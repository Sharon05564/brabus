// AnalyzeView.swift
// Brabus — On-Device Document Intelligence Assistant
//
// Main analysis screen.  The user pastes or types document text, taps
// "Analyze Document", and sees structured results in cards below.
//
// All inference is performed on-device via MLModelManager (Core ML).
// No text is sent to any external service.

import SwiftUI

struct AnalyzeView: View {

    @EnvironmentObject private var historyStore: HistoryStore

    @State private var documentText = ""
    @State private var analysisResult: DocumentAnalysisResult?
    @State private var isAnalyzing = false
    @State private var showEmptyWarning = false

    private let modelManager = MLModelManager()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    inputSection
                    analyzeButton

                    if let result = analysisResult {
                        resultSection(result)
                    } else if !isAnalyzing {
                        emptyState
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 20)
            }
            .navigationTitle("Analyze")
            .navigationBarTitleDisplayMode(.large)
            .background(Color(.systemGroupedBackground))
            .alert("Empty Document", isPresented: $showEmptyWarning) {
                Button("OK", role: .cancel) {}
            } message: {
                Text("Please enter or paste some document text before analyzing.")
            }
            // Pick up text forwarded from ScanView's OCR pipeline and
            // auto-run the analysis so the user sees results immediately.
            .onChange(of: historyStore.pendingText) { _, pending in
                guard !pending.isEmpty else { return }
                documentText = pending
                historyStore.pendingText = ""   // consume so it doesn't re-trigger
                runAnalysis()
            }
        }
    }

    // MARK: - Input section

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionHeader(title: "Document Text")
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(.secondarySystemGroupedBackground))

                if documentText.isEmpty {
                    Text("Paste or type document text here…\n\nExamples: syllabus, receipt, job posting, scholarship notice")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                        .padding(12)
                        .allowsHitTesting(false)
                }

                TextEditor(text: $documentText)
                    .font(.body)
                    .frame(minHeight: 160)
                    .padding(8)
                    .background(.clear)
                    .scrollContentBackground(.hidden)
            }
            .frame(minHeight: 160)
            .clipShape(RoundedRectangle(cornerRadius: 12))

            if !documentText.isEmpty {
                HStack {
                    Text("\(documentText.split(whereSeparator: \.isWhitespace).count) words")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Clear") {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            documentText = ""
                            analysisResult = nil
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(.red)
                }
            }
        }
    }

    // MARK: - Analyze button

    private var analyzeButton: some View {
        Button {
            runAnalysis()
        } label: {
            HStack(spacing: 10) {
                if isAnalyzing {
                    ProgressView()
                        .tint(.white)
                        .scaleEffect(0.9)
                } else {
                    Image(systemName: "brain.head.profile")
                }
                Text(isAnalyzing ? "Analyzing…" : "Analyze Document")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(documentText.isEmpty ? Color.secondary : Color.indigo)
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
        .disabled(isAnalyzing)
    }

    // MARK: - Results

    @ViewBuilder
    private func resultSection(_ result: DocumentAnalysisResult) -> some View {
        VStack(spacing: 16) {
            documentTypeCard(result)
            entitiesCard(result.entities)
            actionCard(result.suggestedAction)
        }
    }

    private func documentTypeCard(_ result: DocumentAnalysisResult) -> some View {
        let docType = result.prediction.documentType
        return VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "Classification Result")
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 14) {
                    ZStack {
                        Circle()
                            .fill(accentColor(for: docType).opacity(0.15))
                            .frame(width: 52, height: 52)
                        Image(systemName: docType.systemIcon)
                            .font(.title3)
                            .foregroundStyle(accentColor(for: docType))
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text(docType.displayName)
                            .font(.title3)
                            .fontWeight(.bold)
                        if result.prediction.confidence >= 0 {
                            Text("Confidence: \(result.prediction.confidenceDescription)")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        } else {
                            Text("Heuristic mode (add .mlmodel for confidence)")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                    }
                    Spacer()
                }
                .padding(16)
            }
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }

    private func entitiesCard(_ entities: ExtractedEntities) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "Extracted Details")
            VStack(spacing: 1) {
                if !entities.dates.isEmpty {
                    EntityRow(icon: "calendar", color: .blue,
                              label: "Dates", values: entities.dates)
                }
                if !entities.emails.isEmpty {
                    EntityRow(icon: "envelope.fill", color: .indigo,
                              label: "Emails", values: entities.emails)
                }
                if !entities.phoneNumbers.isEmpty {
                    EntityRow(icon: "phone.fill", color: .green,
                              label: "Phone Numbers", values: entities.phoneNumbers)
                }
                if !entities.dollarAmounts.isEmpty {
                    EntityRow(icon: "dollarsign.circle.fill", color: .teal,
                              label: "Amounts", values: entities.dollarAmounts)
                }
                if !entities.actionWords.isEmpty {
                    EntityRow(icon: "bolt.fill", color: .orange,
                              label: "Action Words", values: entities.actionWords,
                              isLast: true)
                }
                if entities.isEmpty {
                    HStack {
                        Image(systemName: "info.circle")
                            .foregroundStyle(.secondary)
                        Text("No structured entities detected.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    .padding(16)
                }
            }
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }

    private func actionCard(_ action: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "Suggested Next Action")
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: "arrow.right.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.indigo)
                    .padding(.top, 1)
                Text(action)
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .lineSpacing(4)
                Spacer()
            }
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }

    // MARK: - Empty state

    private var emptyState: some View {
        VStack(spacing: 14) {
            Image(systemName: "doc.text.magnifyingglass")
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text("Ready to Analyze")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Paste document text above and tap \"Analyze Document\" to classify it on-device.")
                .font(.subheadline)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(maxWidth: .infinity)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - Analysis logic

    private func runAnalysis() {
        let trimmed = documentText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            showEmptyWarning = true
            return
        }

        isAnalyzing = true

        // Dispatch to a background thread to avoid blocking the main thread.
        // Core ML inference is fast on modern iPhones, but it's good practice.
        Task.detached(priority: .userInitiated) {
            let prediction = modelManager.predict(text: trimmed)
            let entities = DocumentFeatureExtractor.extractEntities(from: trimmed)
            let suggestion = RecommendationEngine.recommend(
                for: prediction.documentType,
                entities: entities
            )
            let result = DocumentAnalysisResult(
                originalText: trimmed,
                prediction: prediction,
                entities: entities,
                suggestedAction: suggestion
            )

            await MainActor.run {
                withAnimation(.spring(response: 0.4)) {
                    self.analysisResult = result
                    self.isAnalyzing = false
                }
                historyStore.append(result)
            }
        }
    }

    // MARK: - Helpers

    private func accentColor(for type: DocumentType) -> Color {
        switch type {
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

// MARK: - Entity Row

private struct EntityRow: View {
    let icon: String
    let color: Color
    let label: String
    let values: [String]
    var isLast: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: icon)
                    .font(.body)
                    .foregroundStyle(color)
                    .frame(width: 22)
                    .padding(.top, 1)
                VStack(alignment: .leading, spacing: 4) {
                    Text(label)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fontWeight(.medium)
                    FlowLayout(items: values) { value in
                        Text(value)
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(color.opacity(0.12))
                            .foregroundStyle(color)
                            .clipShape(Capsule())
                    }
                }
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)

            if !isLast {
                Divider().padding(.leading, 52)
            }
        }
    }
}

// MARK: - Simple Flow Layout (tag cloud)

private struct FlowLayout<Item: Hashable, Content: View>: View {
    let items: [Item]
    let content: (Item) -> Content
    @State private var totalHeight: CGFloat = 0

    var body: some View {
        GeometryReader { geo in
            self.buildBody(in: geo.size.width)
        }
        .frame(height: totalHeight)
    }

    private func buildBody(in width: CGFloat) -> some View {
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        let itemSpacingH: CGFloat = 6
        let itemSpacingV: CGFloat = 6

        return ZStack(alignment: .topLeading) {
            ForEach(items, id: \.self) { item in
                content(item)
                    .fixedSize()
                    .alignmentGuide(.leading) { d in
                        if currentX + d.width > width {
                            currentX = 0
                            currentY += d.height + itemSpacingV
                        }
                        let offset = currentX
                        currentX += d.width + itemSpacingH
                        return -offset
                    }
                    .alignmentGuide(.top) { _ in -currentY }
            }
        }
        .background(
            GeometryReader { geo in
                Color.clear.onAppear { totalHeight = geo.size.height }
                    .onChange(of: items.count) { _, _ in totalHeight = geo.size.height }
            }
        )
    }
}

#Preview {
    AnalyzeView()
        .environmentObject(HistoryStore())
}
