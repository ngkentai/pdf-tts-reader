import SwiftUI

struct SettingsView: View {
    @Binding var serverURL: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("http://192.168.1.121:8080", text: $serverURL)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Server URL")
                } footer: {
                    Text("Address of the computer serving the pdf_tts folder. Start it with:\n\ncd ~/agents/pdf_tts && python3 -m http.server 8080")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
