// DocumentFeatureExtractor.swift
// Brabus — On-Device Document Intelligence Assistant
//
// Mirrors the Python feature extractor in ml/extract_features.py.
//
// CRITICAL: Feature names, keywords, and regex patterns must stay in sync
// with extract_features.py so the Core ML model receives the same values
// it was trained on.
//
// Feature order (matches FEATURE_ORDER in extract_features.py):
//   0  characterCount
//   1  wordCount
//   2  sentenceCount
//   3  dateCount
//   4  emailCount
//   5  phoneCount
//   6  dollarAmountCount
//   7  actionWordCount
//   8  educationKeywordCount
//   9  jobKeywordCount
//  10  receiptKeywordCount
//  11  deadlineKeywordCount

import Foundation

// MARK: - Keyword Lists

private let actionWords: Set<String> = [
    "submit", "apply", "attend", "register", "pay", "email",
    "upload", "bring", "complete", "review", "contact", "schedule",
    "confirm", "rsvp", "download", "print", "sign", "renew",
    "prepare", "deliver", "return", "visit", "follow", "update",
]

private let educationKeywords: Set<String> = [
    "syllabus", "lecture", "exam", "midterm", "final", "assignment",
    "homework", "quiz", "grade", "professor", "instructor", "course",
    "credit", "semester", "academic", "university", "college", "student",
    "rubric", "transcript", "gpa", "attendance", "textbook", "tuition",
    "enrollment", "prerequisite", "capstone", "thesis", "dissertation",
]

private let jobKeywords: Set<String> = [
    "resume", "internship", "position", "qualifications", "responsibilities",
    "salary", "compensation", "hiring", "candidate", "interview",
    "application", "employer", "full-time", "part-time", "career",
    "skills", "experience", "role", "benefits", "recruiter",
    "cover letter", "linkedin", "job", "opening", "opportunity",
]

private let receiptKeywords: Set<String> = [
    "total", "subtotal", "tax", "payment", "paid", "cash", "credit",
    "debit", "visa", "mastercard", "transaction", "receipt", "purchase",
    "order", "item", "price", "amount", "balance", "refund", "change",
    "invoice", "billing", "charge", "fee", "cost",
]

private let deadlineKeywords: Set<String> = [
    "deadline", "due", "by", "submit by", "no later than", "closes",
    "last day", "final date", "cutoff", "expires", "must be received",
    "applications close", "submission deadline",
]

// MARK: - Regex Patterns

private enum Patterns {
    // Month-name date: "March 15" or "March 15, 2024"
    static let monthNameDate = try! NSRegularExpression(
        pattern: #"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b"#,
        options: .caseInsensitive
    )
    // Numeric date: 03/15/2024 or 2024-03-15
    static let numericDate = try! NSRegularExpression(
        pattern: #"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"#,
        options: []
    )
    // Email
    static let email = try! NSRegularExpression(
        pattern: #"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"#,
        options: []
    )
    // Phone: (404) 555-0192, 404-555-0192, 4045550192
    static let phone = try! NSRegularExpression(
        pattern: #"\b\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"#,
        options: []
    )
    // Dollar amount: $5,000  $89.99
    static let dollar = try! NSRegularExpression(
        pattern: #"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?"#,
        options: []
    )
}

// MARK: - Feature Extractor

struct DocumentFeatureExtractor {

    // MARK: Count helpers

    static func characterCount(_ text: String) -> Int {
        text.count
    }

    static func wordCount(_ text: String) -> Int {
        text.split(whereSeparator: \.isWhitespace).count
    }

    static func sentenceCount(_ text: String) -> Int {
        // Split on . ! ? followed by whitespace or end
        let components = text.components(separatedBy: .init(charactersIn: ".!?"))
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return max(1, components.count)
    }

    static func dateCount(_ text: String) -> Int {
        let range = NSRange(text.startIndex..., in: text)
        let a = Patterns.monthNameDate.numberOfMatches(in: text, range: range)
        let b = Patterns.numericDate.numberOfMatches(in: text, range: range)
        return a + b
    }

    static func emailCount(_ text: String) -> Int {
        let range = NSRange(text.startIndex..., in: text)
        return Patterns.email.numberOfMatches(in: text, range: range)
    }

    static func phoneCount(_ text: String) -> Int {
        let range = NSRange(text.startIndex..., in: text)
        return Patterns.phone.numberOfMatches(in: text, range: range)
    }

    static func dollarAmountCount(_ text: String) -> Int {
        let range = NSRange(text.startIndex..., in: text)
        return Patterns.dollar.numberOfMatches(in: text, range: range)
    }

    static func actionWordCount(_ text: String) -> Int {
        countKeywords(in: text, keywords: actionWords)
    }

    static func educationKeywordCount(_ text: String) -> Int {
        countKeywords(in: text, keywords: educationKeywords)
    }

    static func jobKeywordCount(_ text: String) -> Int {
        countKeywords(in: text, keywords: jobKeywords)
    }

    static func receiptKeywordCount(_ text: String) -> Int {
        countKeywords(in: text, keywords: receiptKeywords)
    }

    static func deadlineKeywordCount(_ text: String) -> Int {
        countKeywords(in: text, keywords: deadlineKeywords)
    }

    // MARK: Feature vector

    /// Returns features as an ordered Float array.
    /// Order must match FEATURE_ORDER in extract_features.py.
    static func featureVector(for text: String) -> [Float] {
        [
            Float(characterCount(text)),
            Float(wordCount(text)),
            Float(sentenceCount(text)),
            Float(dateCount(text)),
            Float(emailCount(text)),
            Float(phoneCount(text)),
            Float(dollarAmountCount(text)),
            Float(actionWordCount(text)),
            Float(educationKeywordCount(text)),
            Float(jobKeywordCount(text)),
            Float(receiptKeywordCount(text)),
            Float(deadlineKeywordCount(text)),
        ]
    }

    /// Returns the full ExtractedDocumentFeatures struct.
    static func extractFeatures(from text: String) -> ExtractedDocumentFeatures {
        ExtractedDocumentFeatures(
            characterCount: characterCount(text),
            wordCount: wordCount(text),
            sentenceCount: sentenceCount(text),
            dateCount: dateCount(text),
            emailCount: emailCount(text),
            phoneCount: phoneCount(text),
            dollarAmountCount: dollarAmountCount(text),
            actionWordCount: actionWordCount(text),
            educationKeywordCount: educationKeywordCount(text),
            jobKeywordCount: jobKeywordCount(text),
            receiptKeywordCount: receiptKeywordCount(text),
            deadlineKeywordCount: deadlineKeywordCount(text)
        )
    }

    // MARK: Entity extraction (for UI display)

    static func extractDates(from text: String) -> [String] {
        let range = NSRange(text.startIndex..., in: text)
        var results: [String] = []
        for match in Patterns.monthNameDate.matches(in: text, range: range) {
            if let r = Range(match.range, in: text) { results.append(String(text[r])) }
        }
        for match in Patterns.numericDate.matches(in: text, range: range) {
            if let r = Range(match.range, in: text) { results.append(String(text[r])) }
        }
        return Array(Set(results)).sorted()
    }

    static func extractEmails(from text: String) -> [String] {
        matches(in: text, pattern: Patterns.email)
    }

    static func extractPhoneNumbers(from text: String) -> [String] {
        matches(in: text, pattern: Patterns.phone)
    }

    static func extractDollarAmounts(from text: String) -> [String] {
        matches(in: text, pattern: Patterns.dollar)
    }

    static func extractActionWords(from text: String) -> [String] {
        let lower = text.lowercased()
        return actionWords.filter { lower.contains($0) }.sorted()
    }

    static func extractEntities(from text: String) -> ExtractedEntities {
        ExtractedEntities(
            dates: extractDates(from: text),
            emails: extractEmails(from: text),
            phoneNumbers: extractPhoneNumbers(from: text),
            dollarAmounts: extractDollarAmounts(from: text),
            actionWords: extractActionWords(from: text)
        )
    }

    // MARK: Private helpers

    private static func countKeywords(in text: String, keywords: Set<String>) -> Int {
        let lower = text.lowercased()
        return keywords.reduce(0) { count, kw in count + (lower.contains(kw) ? 1 : 0) }
    }

    private static func matches(in text: String, pattern: NSRegularExpression) -> [String] {
        let range = NSRange(text.startIndex..., in: text)
        return pattern.matches(in: text, range: range).compactMap { match in
            guard let r = Range(match.range, in: text) else { return nil }
            return String(text[r])
        }
    }
}
