// RecommendationEngine.swift
// Brabus — On-Device Document Intelligence Assistant
//
// Converts a predicted document type and extracted entities into a concise,
// human-readable next-action recommendation.
//
// Design principle: rules are deterministic and explainable.
// No AI API is called — the logic is entirely local.

import Foundation

struct RecommendationEngine {

    /// Generates a recommended next action for the given document type and entities.
    static func recommend(
        for documentType: DocumentType,
        entities: ExtractedEntities
    ) -> String {
        switch documentType {

        case .syllabus:
            return syllabusRecommendation(entities: entities)

        case .assignmentSheet:
            return assignmentRecommendation(entities: entities)

        case .eventFlyer:
            return eventRecommendation(entities: entities)

        case .receipt:
            return receiptRecommendation(entities: entities)

        case .scholarshipNotice:
            return scholarshipRecommendation(entities: entities)

        case .jobPosting:
            return jobRecommendation(entities: entities)

        case .generalNotes:
            return generalNotesRecommendation(entities: entities)

        case .unknown:
            return "Review the document and identify any dates, deadlines, or follow-up tasks."
        }
    }

    // MARK: - Per-type recommendations

    private static func syllabusRecommendation(entities: ExtractedEntities) -> String {
        var parts: [String] = [
            "Save important course dates and review the grading breakdown, attendance policy, and late work policy."
        ]
        if !entities.dates.isEmpty {
            let dateList = entities.dates.prefix(3).joined(separator: ", ")
            parts.append("Key dates found: \(dateList).")
        }
        if !entities.emails.isEmpty {
            parts.append("Instructor contact: \(entities.emails.first!).")
        }
        return parts.joined(separator: " ")
    }

    private static func assignmentRecommendation(entities: ExtractedEntities) -> String {
        var parts: [String] = [
            "Check the deadline, rubric, required format, and submission instructions before starting."
        ]
        if !entities.dates.isEmpty {
            parts.append("Deadline detected: \(entities.dates.first!).")
        }
        if !entities.emails.isEmpty {
            parts.append("Submission email: \(entities.emails.first!).")
        }
        let relevantActions = entities.actionWords.filter {
            ["submit", "upload", "email", "complete", "bring"].contains($0)
        }
        if !relevantActions.isEmpty {
            parts.append("Action required: \(relevantActions.joined(separator: ", ")).")
        }
        return parts.joined(separator: " ")
    }

    private static func eventRecommendation(entities: ExtractedEntities) -> String {
        var parts: [String] = [
            "Consider saving the event date, location, and RSVP details to your calendar."
        ]
        if !entities.dates.isEmpty {
            parts.append("Event date: \(entities.dates.first!).")
        }
        if entities.actionWords.contains("rsvp") {
            parts.append("An RSVP may be required — check the flyer for a deadline or contact.")
        }
        if !entities.emails.isEmpty {
            parts.append("Contact: \(entities.emails.first!).")
        }
        return parts.joined(separator: " ")
    }

    private static func receiptRecommendation(entities: ExtractedEntities) -> String {
        var parts: [String] = [
            "Review the total amount, payment method, and store information for your records."
        ]
        if !entities.dollarAmounts.isEmpty {
            parts.append("Amount(s) found: \(entities.dollarAmounts.joined(separator: ", ")).")
        }
        if !entities.dates.isEmpty {
            parts.append("Transaction date: \(entities.dates.first!).")
        }
        parts.append("Keep this receipt if you may need to return the item or file an expense report.")
        return parts.joined(separator: " ")
    }

    private static func scholarshipRecommendation(entities: ExtractedEntities) -> String {
        var parts: [String] = [
            "Check eligibility requirements and prepare your transcript, resume, and recommendation materials before the deadline."
        ]
        if !entities.dates.isEmpty {
            parts.append("Application deadline: \(entities.dates.first!).")
        }
        if !entities.emails.isEmpty {
            parts.append("Submit materials to: \(entities.emails.first!).")
        }
        if !entities.dollarAmounts.isEmpty {
            parts.append("Award amount: \(entities.dollarAmounts.first!).")
        }
        return parts.joined(separator: " ")
    }

    private static func jobRecommendation(entities: ExtractedEntities) -> String {
        var parts: [String] = [
            "Review the required skills and responsibilities, update your resume, and prepare your application materials."
        ]
        if !entities.dates.isEmpty {
            parts.append("Application deadline: \(entities.dates.first!).")
        }
        if !entities.emails.isEmpty {
            parts.append("Send application to: \(entities.emails.first!).")
        }
        let appActions = entities.actionWords.filter {
            ["apply", "submit", "email", "upload"].contains($0)
        }
        if !appActions.isEmpty {
            parts.append("Next step: \(appActions.first!).")
        }
        return parts.joined(separator: " ")
    }

    private static func generalNotesRecommendation(entities: ExtractedEntities) -> String {
        var parts: [String] = [
            "Summarize the main points and identify any follow-up tasks or reminders."
        ]
        if !entities.dates.isEmpty {
            parts.append("Date reference found: \(entities.dates.first!).")
        }
        let todoActions = entities.actionWords.filter {
            ["complete", "review", "contact", "follow", "submit"].contains($0)
        }
        if !todoActions.isEmpty {
            parts.append("Possible to-do: \(todoActions.prefix(2).joined(separator: ", ")).")
        }
        return parts.joined(separator: " ")
    }
}
