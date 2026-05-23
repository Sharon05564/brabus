// AboutView.swift
// Brabus — On-Device Document Intelligence Assistant
//
// Explains the project, tech stack, and limitations.
// Intended to make the project transparent and recruiter-friendly.

import SwiftUI

struct AboutView: View {

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    headerSection
                    howItWorksSection
                    techStackSection
                    privacySection
                    limitationsSection
                    disclaimerSection
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 20)
            }
            .navigationTitle("About")
            .navigationBarTitleDisplayMode(.large)
            .background(Color(.systemGroupedBackground))
        }
    }

    // MARK: - Sections

    private var headerSection: some View {
        VStack(spacing: 10) {
            ZStack {
                Circle()
                    .fill(LinearGradient(
                        colors: [.indigo, .purple],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ))
                    .frame(width: 72, height: 72)
                Image(systemName: "doc.text.magnifyingglass")
                    .font(.system(size: 32, weight: .medium))
                    .foregroundStyle(.white)
            }
            .shadow(color: .indigo.opacity(0.3), radius: 10, y: 4)

            Text("Brabus")
                .font(.title2)
                .fontWeight(.bold)

            Text("On-Device Document Intelligence")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Text("Version 1.0 — Educational Prototype")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
        .padding(20)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var howItWorksSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "How It Works")
            VStack(spacing: 1) {
                StepRow(number: "1", title: "Input",
                        detail: "You scan a photo (Vision OCR) or paste document text.")
                StepRow(number: "2", title: "Feature Extraction",
                        detail: "Brabus counts dates, emails, keywords, and other signals.")
                StepRow(number: "3", title: "On-Device Classification",
                        detail: "A Core ML model — trained in Python — predicts the document type.")
                StepRow(number: "4", title: "Entity Extraction",
                        detail: "Regex patterns pull out dates, emails, amounts, and action words.")
                StepRow(number: "5", title: "Recommendation",
                        detail: "A rule-based engine suggests a concrete next action.", isLast: true)
            }
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }

    private var techStackSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "Tech Stack")
            VStack(spacing: 1) {
                TechRow(icon: "swift", color: .orange, name: "Swift & SwiftUI",
                        detail: "iOS app built with SwiftUI, TabView, Navigation")
                TechRow(icon: "brain.head.profile", color: .indigo, name: "Core ML",
                        detail: "On-device inference — no server required")
                TechRow(icon: "viewfinder", color: .blue, name: "Vision",
                        detail: "VNRecognizeTextRequest for photo OCR")
                TechRow(icon: "flame.fill", color: .red, name: "PyTorch",
                        detail: "DocumentClassifier trained with CrossEntropyLoss + Adam")
                TechRow(icon: "chart.bar.fill", color: .teal, name: "scikit-learn",
                        detail: "TF-IDF + Logistic Regression / LinearSVC baselines")
                TechRow(icon: "tablecells.fill", color: .green, name: "pandas & NumPy",
                        detail: "Data generation, feature matrix construction")
                TechRow(icon: "arrow.triangle.2.circlepath", color: .purple,
                        name: "coremltools", detail: "PyTorch → Core ML conversion",
                        isLast: true)
            }
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }

    private var privacySection: some View {
        HStack(spacing: 14) {
            Image(systemName: "lock.shield.fill")
                .font(.title2)
                .foregroundStyle(.green)
            VStack(alignment: .leading, spacing: 4) {
                Text("Privacy First")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Text("Brabus does not send document text or images to any server. All processing — OCR, feature extraction, and classification — happens entirely on your device.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineSpacing(3)
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var limitationsSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "Known Limitations")
            VStack(spacing: 1) {
                LimitRow(text: "Trained on a synthetic dataset — real-world accuracy may vary.")
                LimitRow(text: "OCR quality depends on image resolution and lighting.")
                LimitRow(text: "Seven document categories only; unusual documents may misclassify.")
                LimitRow(text: "Numeric features alone may not capture all text nuances.", isLast: true)
            }
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }

    private var disclaimerSection: some View {
        Text("Brabus is an educational machine learning prototype built for a portfolio project. It is not intended to provide legal, financial, academic, or professional advice. Predictions may be incorrect. Always verify important information independently.")
            .font(.caption2)
            .foregroundStyle(.tertiary)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
    }
}

// MARK: - Subviews

private struct StepRow: View {
    let number: String
    let title: String
    let detail: String
    var isLast: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .top, spacing: 14) {
                ZStack {
                    Circle()
                        .fill(Color.indigo.opacity(0.12))
                        .frame(width: 28, height: 28)
                    Text(number)
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundStyle(.indigo)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)

            if !isLast { Divider().padding(.leading, 58) }
        }
    }
}

private struct TechRow: View {
    let icon: String
    let color: Color
    let name: String
    let detail: String
    var isLast: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                Image(systemName: icon)
                    .font(.body)
                    .foregroundStyle(color)
                    .frame(width: 24)
                VStack(alignment: .leading, spacing: 2) {
                    Text(name)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 11)

            if !isLast { Divider().padding(.leading, 54) }
        }
    }
}

private struct LimitRow: View {
    let text: String
    var isLast: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "exclamationmark.circle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .padding(.top, 1)
                Text(text)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            if !isLast { Divider().padding(.leading, 40) }
        }
    }
}

#Preview {
    AboutView()
}
