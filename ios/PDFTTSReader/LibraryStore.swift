import Foundation

enum LibraryError: LocalizedError {
    case badURL
    case badResponse(Int)
    case noAudioFound

    var errorDescription: String? {
        switch self {
        case .badURL: return "Invalid server URL"
        case .badResponse(let code): return "Server returned HTTP \(code)"
        case .noAudioFound: return "Could not determine the audio file for this document"
        }
    }
}

@MainActor
final class LibraryStore: ObservableObject {
    @Published var localDocs: [LocalDoc] = []
    @Published var remoteDocs: [RemoteDoc] = []
    @Published var downloadProgress: [String: Double] = [:]   // doc id → 0…1
    @Published var isRefreshing = false
    @Published var lastError: String?
    /// Set once a manifest fetch succeeds or fails, so the UI can distinguish
    /// "not synced yet" from "server has nothing new".
    @Published var hasSynced = false

    private var downloadTasks: [String: Task<Void, Never>] = [:]

    static var docsRoot: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    init() {
        loadLocal()
    }

    // MARK: - Local documents

    func loadLocal() {
        let fm = FileManager.default
        var docs: [LocalDoc] = []
        let folders = (try? fm.contentsOfDirectory(at: Self.docsRoot,
                                                   includingPropertiesForKeys: nil)) ?? []
        for folder in folders {
            let meta = folder.appendingPathComponent("doc.json")
            guard let data = try? Data(contentsOf: meta),
                  let doc = try? JSONDecoder().decode(LocalDoc.self, from: data) else { continue }
            docs.append(doc)
        }
        localDocs = docs.sorted { $0.title < $1.title }
    }

    func folder(for doc: LocalDoc) -> URL {
        Self.docsRoot.appendingPathComponent(doc.id, isDirectory: true)
    }

    func delete(_ doc: LocalDoc) {
        try? FileManager.default.removeItem(at: folder(for: doc))
        UserDefaults.standard.removeObject(forKey: "tts_pos_\(doc.id)")
        loadLocal()
    }

    /// Saved playback position in seconds (same key the viewer uses in localStorage).
    func savedPosition(_ docID: String) -> Double {
        UserDefaults.standard.double(forKey: "tts_pos_\(docID)")
    }

    // MARK: - Server

    private func serverBase(_ serverURL: String) throws -> URL {
        let trimmed = serverURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed), url.scheme != nil else {
            throw LibraryError.badURL
        }
        return url
    }

    func refresh(serverURL: String) async {
        isRefreshing = true
        lastError = nil
        defer { isRefreshing = false; hasSynced = true }
        do {
            let base = try serverBase(serverURL)
            let url = base.appendingPathComponent("manifest.json")
            var request = URLRequest(url: url)
            request.cachePolicy = .reloadIgnoringLocalCacheData
            let (data, response) = try await URLSession.shared.data(for: request)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                throw LibraryError.badResponse(http.statusCode)
            }
            remoteDocs = try JSONDecoder().decode(Manifest.self, from: data).documents
        } catch {
            remoteDocs = []
            lastError = "Could not reach server: \(error.localizedDescription)"
        }
    }

    /// Remote docs not yet downloaded.
    var availableDocs: [RemoteDoc] {
        let localIDs = Set(localDocs.map { $0.id })
        return remoteDocs.filter { !localIDs.contains($0.id) }
    }

    // MARK: - Download

    func startDownload(_ doc: RemoteDoc, serverURL: String) {
        guard downloadTasks[doc.id] == nil else { return }
        downloadProgress[doc.id] = 0
        lastError = nil
        downloadTasks[doc.id] = Task {
            do {
                try await self.download(doc, serverURL: serverURL)
            } catch is CancellationError {
                // user cancelled — nothing to report
            } catch let error as URLError where error.code == .cancelled {
                // user cancelled — nothing to report
            } catch {
                self.lastError = "Download of “\(doc.title)” failed: \(error.localizedDescription)"
            }
            self.downloadProgress[doc.id] = nil
            self.downloadTasks[doc.id] = nil
            self.loadLocal()
        }
    }

    func cancelDownload(_ docID: String) {
        downloadTasks[docID]?.cancel()
    }

    private static func fileSize(at url: URL) -> Int64 {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs?[.size] as? Int64) ?? 0
    }

    private func download(_ doc: RemoteDoc, serverURL: String) async throws {
        let base = try serverBase(serverURL)
        let fm = FileManager.default
        let tmpDir = Self.docsRoot.appendingPathComponent(".tmp_\(doc.id)", isDirectory: true)
        try? fm.removeItem(at: tmpDir)
        try fm.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmpDir) }

        // Weight progress by known sizes; the viewer is small next to the audio.
        let viewerWeight: Double
        if let a = doc.audio_bytes, let v = doc.viewer_bytes, a + v > 0 {
            viewerWeight = Double(v) / Double(a + v)
        } else {
            viewerWeight = 0.05
        }

        let viewerURL = base.appendingPathComponent(doc.viewer)
        let viewerDest = tmpDir.appendingPathComponent("viewer.html")
        try await Downloader.fetch(viewerURL, to: viewerDest) { p in
            Task { @MainActor in self.downloadProgress[doc.id] = p * viewerWeight }
        }
        try Task.checkCancellation()

        // Which audio file does the viewer reference?
        var audioFilename = doc.audio.map { ($0 as NSString).lastPathComponent }
        if audioFilename == nil {
            let head = (try? String(contentsOf: viewerDest, encoding: .utf8))?.prefix(200_000) ?? ""
            if let range = head.range(of: #"<source src="([^"]+)""#, options: .regularExpression) {
                let src = String(head[range].dropFirst(#"<source src=""#.count).dropLast())
                if src.hasPrefix("data:") {
                    audioFilename = ""   // audio embedded in the HTML, nothing to download
                } else {
                    audioFilename = (src as NSString).lastPathComponent
                }
            }
        }
        guard let audioName = audioFilename else { throw LibraryError.noAudioFound }

        var totalBytes = Self.fileSize(at: viewerDest)

        if !audioName.isEmpty {
            let audioURL = base
                .appendingPathComponent(doc.id)
                .appendingPathComponent(audioName)
            let audioDest = tmpDir.appendingPathComponent(audioName)
            try await Downloader.fetch(audioURL, to: audioDest) { p in
                Task { @MainActor in
                    self.downloadProgress[doc.id] = viewerWeight + p * (1 - viewerWeight)
                }
            }
            totalBytes += Self.fileSize(at: audioDest)
        }
        try Task.checkCancellation()

        let local = LocalDoc(id: doc.id, title: doc.title, language: doc.language,
                             durationMin: doc.duration_min, generated: doc.generated,
                             audioFilename: audioName, totalBytes: totalBytes)
        let meta = try JSONEncoder().encode(local)
        try meta.write(to: tmpDir.appendingPathComponent("doc.json"))

        let finalDir = Self.docsRoot.appendingPathComponent(doc.id, isDirectory: true)
        try? fm.removeItem(at: finalDir)
        try fm.moveItem(at: tmpDir, to: finalDir)

        // Downloaded documents should not be purged by the system or backed up.
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var dir = finalDir
        try? dir.setResourceValues(values)
    }
}

/// Progress-reporting file download (URLSession's async APIs have no progress callback).
private enum Downloader {

    static func fetch(_ url: URL, to dest: URL,
                      progress: @escaping (Double) -> Void) async throws {
        let delegate = Delegate(dest: dest, onProgress: progress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                delegate.continuation = cont
                let task = session.downloadTask(with: url)
                delegate.task = task
                task.resume()
            }
        } onCancel: {
            delegate.task?.cancel()
        }
    }

    private final class Delegate: NSObject, URLSessionDownloadDelegate {
        let dest: URL
        let onProgress: (Double) -> Void
        var continuation: CheckedContinuation<Void, Error>?
        var task: URLSessionDownloadTask?
        private var moveError: Error?

        init(dest: URL, onProgress: @escaping (Double) -> Void) {
            self.dest = dest
            self.onProgress = onProgress
        }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                        didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                        totalBytesExpectedToWrite: Int64) {
            guard totalBytesExpectedToWrite > 0 else { return }
            onProgress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
        }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                        didFinishDownloadingTo location: URL) {
            if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
                moveError = LibraryError.badResponse(http.statusCode)
                return
            }
            do {
                try? FileManager.default.removeItem(at: dest)
                try FileManager.default.moveItem(at: location, to: dest)
            } catch {
                moveError = error
            }
        }

        func urlSession(_ session: URLSession, task: URLSessionTask,
                        didCompleteWithError error: Error?) {
            let cont = continuation
            continuation = nil
            if let error {
                cont?.resume(throwing: error)
            } else if let moveError {
                cont?.resume(throwing: moveError)
            } else {
                cont?.resume()
            }
        }
    }
}
