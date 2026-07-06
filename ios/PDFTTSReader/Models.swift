import Foundation

/// Wire format of manifest.json served from the computer (pdf_tts folder).
struct Manifest: Codable {
    let documents: [RemoteDoc]
}

struct RemoteDoc: Codable, Identifiable {
    let id: String
    let title: String
    let language: String?
    let duration_min: Double?
    let viewer: String
    let generated: String?
    /// Server-relative path of the audio file, e.g. "Einfuehrung_tts/audio_64k.mp3".
    /// Older manifests lack it; the app then extracts it from the downloaded viewer.html.
    let audio: String?
    let audio_bytes: Int64?
    let viewer_bytes: Int64?

    var totalBytes: Int64? {
        guard let a = audio_bytes else { return nil }
        return a + (viewer_bytes ?? 0)
    }
}

/// A document downloaded to the phone. Persisted as doc.json inside its folder.
struct LocalDoc: Codable, Identifiable {
    let id: String
    let title: String
    let language: String?
    let durationMin: Double?
    let generated: String?
    let audioFilename: String
    let totalBytes: Int64
}

enum Fmt {
    static func bytes(_ n: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: n, countStyle: .file)
    }

    static func duration(_ min: Double) -> String {
        if min >= 60 {
            let h = Int(min) / 60
            let m = Int(min.rounded()) % 60
            return m > 0 ? "\(h)h \(m)m" : "\(h)h"
        }
        return "\(Int(min.rounded())) min"
    }

    static func clock(_ secs: Double) -> String {
        let s = max(0, Int(secs))
        if s >= 3600 {
            return String(format: "%d:%02d:%02d", s / 3600, (s % 3600) / 60, s % 60)
        }
        return String(format: "%d:%02d", s / 60, s % 60)
    }

    static func flag(_ language: String?) -> String {
        switch language?.lowercased() {
        case "en": return "🇺🇸"
        case "en-gb": return "🇬🇧"
        case "de": return "🇩🇪"
        case "fr": return "🇫🇷"
        case "es": return "🇪🇸"
        case "it": return "🇮🇹"
        case "ja": return "🇯🇵"
        case "zh": return "🇨🇳"
        case "pt": return "🇧🇷"
        case "ko": return "🇰🇷"
        case "nl": return "🇳🇱"
        default: return "📄"
        }
    }
}
