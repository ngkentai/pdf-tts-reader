import SwiftUI
import WebKit
import AVFoundation

/// Renders a downloaded viewer.html in a WKWebView. All word-sync logic lives
/// in the viewer itself; this wrapper only supplies the saved playback position
/// and persists new positions natively (localStorage for file:// pages is not
/// reliably persisted across launches).
struct ReaderView: View {
    let doc: LocalDoc
    @EnvironmentObject private var library: LibraryStore

    var body: some View {
        ViewerWebView(doc: doc, folder: library.folder(for: doc))
            .ignoresSafeArea(edges: .bottom)
            .navigationTitle(doc.title)
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                try? AVAudioSession.sharedInstance().setActive(true)
                UIApplication.shared.isIdleTimerDisabled = true
            }
            .onDisappear {
                UIApplication.shared.isIdleTimerDisabled = false
            }
    }
}

private struct ViewerWebView: UIViewRepresentable {
    let doc: LocalDoc
    let folder: URL

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []
        config.allowsAirPlayForMediaPlayback = true

        let posKey = "tts_pos_\(doc.id)"
        let saved = UserDefaults.standard.double(forKey: posKey)
        // Seed the saved position before the viewer script reads localStorage,
        // and mirror every position write back to the app.
        let bridge = """
        (function() {
          try {
            if (\(saved) > 1) localStorage.setItem('\(posKey)', '\(saved)');
            var orig = Storage.prototype.setItem;
            Storage.prototype.setItem = function(k, v) {
              orig.call(this, k, v);
              if (String(k).indexOf('tts_pos_') === 0) {
                try { window.webkit.messageHandlers.pos.postMessage({key: k, value: v}); } catch (e) {}
              }
            };
          } catch (e) {}
        })();
        """
        config.userContentController.addUserScript(
            WKUserScript(source: bridge, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        config.userContentController.add(context.coordinator, name: "pos")

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.isOpaque = false
        webView.backgroundColor = UIColor(red: 0.98, green: 0.976, blue: 0.968, alpha: 1)
        webView.allowsBackForwardNavigationGestures = false

        let viewer = folder.appendingPathComponent("viewer.html")
        webView.loadFileURL(viewer, allowingReadAccessTo: folder)
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {}

    static func dismantleUIView(_ webView: WKWebView, coordinator: Coordinator) {
        webView.configuration.userContentController.removeScriptMessageHandler(forName: "pos")
    }

    final class Coordinator: NSObject, WKScriptMessageHandler {
        func userContentController(_ userContentController: WKUserContentController,
                                   didReceive message: WKScriptMessage) {
            guard message.name == "pos",
                  let body = message.body as? [String: Any],
                  let key = body["key"] as? String,
                  let value = body["value"] as? String,
                  let pos = Double(value) else { return }
            UserDefaults.standard.set(pos, forKey: key)
        }
    }
}
