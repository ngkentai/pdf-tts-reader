import SwiftUI
import AVFoundation

@main
struct PDFTTSReaderApp: App {
    @StateObject private var library = LibraryStore()

    init() {
        // .playback + the "audio" background mode keeps the WKWebView's
        // <audio> element playing when the screen locks.
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio)
    }

    var body: some Scene {
        WindowGroup {
            LibraryView()
                .environmentObject(library)
        }
    }
}
