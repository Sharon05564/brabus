// DocumentModels.swift
// Brabus — On-Device Document Intelligence Assistant
//
// Shared Swift types used across the entire app.
// Keep this file as the single source of truth for all data shapes.

import Foundation

// MARK: - Document Type Enum

enum DocumentType: String, CaseIterable, Codable {
    case syllabus          = "syllabus"
    case assignmentSheet   = "assignment_sheet"
    case eventFlyer        = "event_flyer"
    case receipt           = "receipt"
    case scholarshipNotice = "scholarship_notice"
    case jobPosting        = "job_posting"
    case generalNotes      = "general_notes"
    case unknown           = "unknown"

    var displayName: String {
        switch self {
        case .syllabus:          return "Syllabus"
        case .assignmentSheet:   return "Assignment Sheet"
        case .eventFlyer:        return "Event Flyer"
        case .receipt:           return "Receipt"
        case .scholarshipNotice: return "Scholarship Notice"
        case .jobPosting:        return "Job Posting"
        case .generalNotes:      return "General Notes"
        case .unknown:           return "Unknown"
        }
    }

    var systemIcon: String {
        switch self {
        case .syllabus:          return "book.fill"
        case .assignmentSheet:   return "doc.text.fill"
        case .eventFlyer:        return "calendar.badge.plus"
        case .receipt:           return "receipt.fill"
        case .scholarshipNotice: return "graduationcap.fill"
        case .jobPosting:        return "briefcase.fill"
        case .generalNotes:      return "note.text"
        case .unknown:           return "questionmark.circle.fill"
        }
    }

    var accentColorName: String {
        switch self {
        case .syllabus:          return "blue"
        case .assignmentSheet:   return "orange"
        case .eventFlyer:        return "purple"
        case .receipt:           return "green"
        case .scholarshipNotice: return "indigo"
        case .jobPosting:        return "teal"
        case .generalNotes:      return "gray"
        case .unknown:           return "gray"
        }
    }

    static func from(rawValue: String) -> DocumentType {
        return DocumentType(rawValue: rawValue) ?? .unknown
    }
}

// MARK: - Extracted Document Features (numeric)

/// Holds the 12 numeric features extracted from a document.
/// Field order matches FEATURE_ORDER in extract_features.py and
/// DocumentFeatureExtractor.swift.
struct ExtractedDocumentFeatures {
    let characterCount: Int
    let wordCount: Int
    let sentenceCount: Int
    let dateCount: Int
    let emailCount: Int
    let phoneCount: Int
    let dollarAmountCount: Int
    let actionWordCount: Int
    let educationKeywordCount: Int
    let jobKeywordCount: Int
    let receiptKeywordCount: Int
    let deadlineKeywordCount: Int

    /// Returns features as an ordered Float array for the Core ML model.
    /// Must match the order in extract_features.py > FEATURE_ORDER.
    var asFloatArray: [Float] {
        [
            Float(characterCount),
            Float(wordCount),
            Float(sentenceCount),
            Float(dateCount),
            Float(emailCount),
            Float(phoneCount),
            Float(dollarAmountCount),
            Float(actionWordCount),
            Float(educationKeywordCount),
            Float(jobKeywordCount),
            Float(receiptKeywordCount),
            Float(deadlineKeywordCount),
        ]
    }
}

// MARK: - Extracted Text Entities

/// Holds entity strings extracted from the document text (emails, dates, etc.)
struct ExtractedEntities {
    let dates: [String]
    let emails: [String]
    let phoneNumbers: [String]
    let dollarAmounts: [String]
    let actionWords: [String]

    var isEmpty: Bool {
        dates.isEmpty &&
        emails.isEmpty &&
        phoneNumbers.isEmpty &&
        dollarAmounts.isEmpty &&
        actionWords.isEmpty
    }
}

// MARK: - Prediction Result

/// The output of MLModelManager after running inference.
struct PredictionResult {
    let documentType: DocumentType
    let confidence: Double        // 0.0–1.0; -1 if unavailable
    let isConfident: Bool         // true when confidence ≥ 0.60

    static let unavailable = PredictionResult(
        documentType: .unknown,
        confidence: -1,
        isConfident: false
    )

    var confidenceDescription: String {
        guard confidence >= 0 else { return "" }
        return String(format: "%.0f%%", confidence * 100)
    }
}

// MARK: - Full Analysis Result

/// Everything Brabus knows about a scanned or pasted document.
/// Stored in HistoryView and displayed in AnalyzeView.
struct DocumentAnalysisResult: Identifiable {
    let id: UUID
    let timestamp: Date
    let originalText: String
    let prediction: PredictionResult
    let entities: ExtractedEntities
    let suggestedAction: String

    init(
        originalText: String,
        prediction: PredictionResult,
        entities: ExtractedEntities,
        suggestedAction: String
    ) {
        self.id = UUID()
        self.timestamp = Date()
        self.originalText = originalText
        self.prediction = prediction
        self.entities = entities
        self.suggestedAction = suggestedAction
    }

    var textPreview: String {
        let trimmed = originalText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > 80 else { return trimmed }
        return String(trimmed.prefix(80)) + "…"
    }

    var formattedTimestamp: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: timestamp)
    }
}
