import SwiftUI

struct LibraryView: View {
    @EnvironmentObject private var library: LibraryStore
    @AppStorage("serverURL") private var serverURL = "http://192.168.1.121:8080"
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            List {
                downloadedSection
                availableSection
                if let error = library.lastError {
                    Section {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .font(.footnote)
                            .foregroundStyle(.red)
                    }
                }
            }
            .navigationDestination(for: String.self) { docID in
                if let doc = library.localDocs.first(where: { $0.id == docID }) {
                    ReaderView(doc: doc)
                }
            }
            .navigationTitle("TTS Library")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { showSettings = true } label: {
                        Image(systemName: "gearshape")
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await library.refresh(serverURL: serverURL) }
                    } label: {
                        if library.isRefreshing {
                            ProgressView()
                        } else {
                            Image(systemName: "arrow.triangle.2.circlepath")
                        }
                    }
                    .disabled(library.isRefreshing)
                }
            }
            .refreshable { await library.refresh(serverURL: serverURL) }
            .sheet(isPresented: $showSettings) {
                SettingsView(serverURL: $serverURL)
            }
            .task {
                if !library.hasSynced {
                    await library.refresh(serverURL: serverURL)
                }
            }
        }
    }

    // MARK: - Sections

    private var downloadedSection: some View {
        Section("On this iPhone") {
            if library.localDocs.isEmpty {
                Text("No documents downloaded yet. Sync with your computer below.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            ForEach(library.localDocs) { doc in
                NavigationLink(value: doc.id) {
                    LocalDocRow(doc: doc)
                }
                .swipeActions {
                    Button(role: .destructive) {
                        library.delete(doc)
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var availableSection: some View {
        Section("On the computer") {
            if !library.hasSynced && !library.isRefreshing {
                Text("Pull down or tap sync to list documents on \(serverURL).")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else if library.hasSynced && library.availableDocs.isEmpty && library.lastError == nil {
                Text("Everything on the server is already downloaded.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            ForEach(library.availableDocs) { doc in
                RemoteDocRow(doc: doc, serverURL: serverURL)
            }
        }
    }
}

// MARK: - Rows

private struct LocalDocRow: View {
    let doc: LocalDoc
    @EnvironmentObject private var library: LibraryStore

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 8) {
                Text(Fmt.flag(doc.language))
                VStack(alignment: .leading, spacing: 2) {
                    Text(doc.title)
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(2)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            if let totalSecs = doc.durationMin.map({ $0 * 60 }), totalSecs > 0 {
                let pos = library.savedPosition(doc.id)
                ProgressView(value: min(pos, totalSecs), total: totalSecs)
                    .tint(pos / totalSecs >= 0.98 ? .green : .accentColor)
            }
        }
        .padding(.vertical, 2)
    }

    private var subtitle: String {
        var parts: [String] = []
        if let min = doc.durationMin {
            let pos = library.savedPosition(doc.id)
            if pos > 30 {
                parts.append("\(Fmt.clock(pos)) of \(Fmt.duration(min))")
            } else {
                parts.append(Fmt.duration(min))
            }
        }
        parts.append(Fmt.bytes(doc.totalBytes))
        return parts.joined(separator: " · ")
    }
}

private struct RemoteDocRow: View {
    let doc: RemoteDoc
    let serverURL: String
    @EnvironmentObject private var library: LibraryStore

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            Text(Fmt.flag(doc.language))
            VStack(alignment: .leading, spacing: 2) {
                Text(doc.title)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(2)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if let progress = library.downloadProgress[doc.id] {
                Button {
                    library.cancelDownload(doc.id)
                } label: {
                    ZStack {
                        Circle()
                            .stroke(Color.secondary.opacity(0.25), lineWidth: 3)
                        Circle()
                            .trim(from: 0, to: max(0.02, progress))
                            .stroke(Color.accentColor,
                                    style: StrokeStyle(lineWidth: 3, lineCap: .round))
                            .rotationEffect(.degrees(-90))
                        Image(systemName: "stop.fill")
                            .font(.system(size: 8))
                            .foregroundStyle(Color.accentColor)
                    }
                    .frame(width: 26, height: 26)
                }
                .buttonStyle(.plain)
            } else {
                Button {
                    library.startDownload(doc, serverURL: serverURL)
                } label: {
                    Image(systemName: "arrow.down.circle")
                        .font(.title3)
                }
                .buttonStyle(.plain)
                .foregroundStyle(Color.accentColor)
            }
        }
        .padding(.vertical, 2)
    }

    private var subtitle: String {
        var parts: [String] = []
        if let min = doc.duration_min { parts.append(Fmt.duration(min)) }
        if let total = doc.totalBytes { parts.append(Fmt.bytes(total)) }
        if let gen = doc.generated { parts.append(gen) }
        return parts.joined(separator: " · ")
    }
}
