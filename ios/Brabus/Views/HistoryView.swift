// HistoryView.swift
// Brabus — On-Device Document Intelligence Assistant
//
// Displays all analysis results from the current session in a scrollable
// card list. Tapping a result shows its full details.

import SwiftUI

struct HistoryView: View {

    @EnvironmentObject private var historyStore: HistoryStore
    @State private var selectedResult: DocumentAnalysisResult?
    @State private var showClearConfirm = false

    var body: some View {
        NavigationStack {
            Group {
                if historyStore.results.isEmpty {
                    emptyState
                } else {
                    resultList
                }
            }
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.large)
            .background(Color(.systemGroupedBackground))
            .toolbar {
                if !historyStore.results.isEmpty {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Clear") { showClearConfirm = true }
                            .foregroundStyle(.red)
                    }
                }
            }
            .confirmationDialog(
                "Clear all history?",
                isPresented: $showClearConfirm,
                titleVisibility: .visible
            ) {
                Button("Clear All", role: .destructive) { historyStore.clear() }
                Button("Cancel", role: .cancel) {}
            }
            .sheet(item: $selectedResult) { result in
                HistoryDetailSheet(result: result)
            }
        }
    }

    // MARK: - List

    private var resultList: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                ForEach(historyStore.results) { result in
                    HistoryCard(result: result)
                        .onTapGesture { selectedResult = result }
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
        }
    }

    // MARK: - Empty state

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "clock.badge.questionmark")
                .font(.system(size: 52))
                .foregroundStyle(.tertiary)
            Text("No History Yet")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Analyzed documents will appear here during your current session.")
                .font(.subheadline)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - History Card

private struct HistoryCard: View {
    let result: DocumentAnalysisResult

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                // Document type icon
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(typeColor.opacity(0.12))
                        .frame(width: 44, height: 44)
                    Image(systemName: result.prediction.documentType.systemIcon)
                        .font(.body)
                        .foregroundStyle(typeColor)
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text(result.prediction.documentType.displayName)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    Text(result.formattedTimestamp)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }

            Text(result.textPreview)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .lineSpacing(3)

            if !result.suggestedAction.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "arrow.right.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.indigo)
                        .padding(.top, 1)
                    Text(result.suggestedAction)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                .padding(10)
                .background(Color.indigo.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var typeColor: Color {
        switch result.prediction.documentType {
        case .syllabus:          return .blue
        case .assignmentSheet:   return .orange
        case .eventFlyer:        return .purple
        case .receipt:           return .green
        case .scholarshipNotice: return .indigo
        case .jobPosting:        return .teal
        case .generalNotes:      return .gray
        case .unknown:           return .gray
        }
    }
}

// MARK: - History Detail Sheet

struct HistoryDetailSheet: View {
    let result: DocumentAnalysisResult
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    typeSection
                    if !result.entities.isEmpty {
                        entitiesSummarySection
                    }
                    actionSection
                    textSection
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 20)
            }
            .navigationTitle(result.prediction.documentType.displayName)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .background(Color(.systemGroupedBackground))
        }
    }

    private var typeSection: some View {
        HStack(spacing: 14) {
            Image(systemName: result.prediction.documentType.systemIcon)
                .font(.title2)
                .foregroundStyle(.indigo)
            VStack(alignment: .leading, spacing: 3) {
                Text(result.prediction.documentType.displayName)
                    .font(.headline)
                Text(result.formattedTimestamp)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if result.prediction.confidence >= 0 {
                    Text("Confidence: \(result.prediction.confidenceDescription)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var entitiesSummarySection: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "Detected Details")
            VStack(alignment: .leading, spacing: 8) {
                EntityLine(label: "Dates", values: result.entities.dates)
                EntityLine(label: "Emails", values: result.entities.emails)
                EntityLine(label: "Phone Numbers", values: result.entities.phoneNumbers)
                EntityLine(label: "Amounts", values: result.entities.dollarAmounts)
                EntityLine(label: "Action Words", values: result.entities.actionWords)
            }
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }

    private var actionSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "Suggested Action")
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "arrow.right.circle.fill")
                    .foregroundStyle(.indigo)
                Text(result.suggestedAction)
                    .font(.subheadline)
                    .lineSpacing(4)
                Spacer()
            }
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }

    private var textSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "Original Text")
            ScrollView {
                Text(result.originalText)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(14)
            }
            .frame(maxHeight: 180)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }
}

private struct EntityLine: View {
    let label: String
    let values: [String]

    var body: some View {
        if !values.isEmpty {
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.caption2)
                    .fontWeight(.semibold)
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                Text(values.joined(separator: "  ·  "))
                    .font(.caption)
                    .foregroundStyle(.primary)
            }
        }
    }
}

#Preview {
    HistoryView()
        .environmentObject(HistoryStore())
}
