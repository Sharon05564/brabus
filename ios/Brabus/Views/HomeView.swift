// HomeView.swift
// Brabus — On-Device Document Intelligence Assistant
//
// Landing screen: project name, description, quick-action buttons,
// and a privacy notice.

import SwiftUI

struct HomeView: View {

    @Binding var selectedTab: ContentView.Tab

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 32) {
                    heroSection
                    quickActionsSection
                    featureHighlightsSection
                    privacyBadge
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 24)
            }
            .navigationTitle("")
            .navigationBarHidden(true)
            .background(Color(.systemGroupedBackground))
        }
    }

    // MARK: - Hero

    private var heroSection: some View {
        VStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [.indigo, .purple],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 88, height: 88)
                Image(systemName: "doc.text.magnifyingglass")
                    .font(.system(size: 40, weight: .medium))
                    .foregroundStyle(.white)
            }
            .shadow(color: .indigo.opacity(0.35), radius: 12, y: 6)

            Text("Brabus")
                .font(.system(size: 36, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            Text("Brabus turns scanned documents into structured,\nuseful actions using on-device intelligence.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .lineSpacing(4)
        }
        .padding(.top, 12)
    }

    // MARK: - Quick Actions

    private var quickActionsSection: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                ActionButton(
                    title: "Scan Document",
                    subtitle: "Use Vision OCR",
                    icon: "viewfinder.circle.fill",
                    color: .indigo
                ) {
                    selectedTab = .scan
                }

                ActionButton(
                    title: "Paste Text",
                    subtitle: "Type or paste",
                    icon: "text.cursor",
                    color: .purple
                ) {
                    selectedTab = .analyze
                }
            }
        }
    }

    // MARK: - Feature Highlights

    private var featureHighlightsSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "What Brabus Does")

            VStack(spacing: 1) {
                FeatureRow(
                    icon: "brain.head.profile",
                    color: .indigo,
                    title: "On-Device Classification",
                    detail: "Identifies 7 document types using a Core ML model"
                )
                FeatureRow(
                    icon: "text.magnifyingglass",
                    color: .purple,
                    title: "Entity Extraction",
                    detail: "Pulls out dates, emails, deadlines, and amounts"
                )
                FeatureRow(
                    icon: "arrow.right.circle.fill",
                    color: .teal,
                    title: "Action Recommendations",
                    detail: "Suggests a concrete next step for each document"
                )
                FeatureRow(
                    icon: "viewfinder",
                    color: .orange,
                    title: "Vision OCR",
                    detail: "Reads text from photos using Apple's Vision framework"
                )
                FeatureRow(
                    icon: "clock.fill",
                    color: .gray,
                    title: "Session History",
                    detail: "Keeps a list of analyses during your current session",
                    isLast: true
                )
            }
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }

    // MARK: - Privacy Badge

    private var privacyBadge: some View {
        HStack(spacing: 10) {
            Image(systemName: "lock.shield.fill")
                .foregroundStyle(.green)
                .font(.title3)
            VStack(alignment: .leading, spacing: 2) {
                Text("Privacy Protected")
                    .font(.footnote)
                    .fontWeight(.semibold)
                    .foregroundStyle(.primary)
                Text("No document text is sent to a server.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(14)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }
}

// MARK: - Subviews

private struct ActionButton: View {
    let title: String
    let subtitle: String
    let icon: String
    let color: Color
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 28))
                    .foregroundStyle(color)
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(.primary)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 20)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }
}

private struct FeatureRow: View {
    let icon: String
    let color: Color
    let title: String
    let detail: String
    var isLast: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                Image(systemName: icon)
                    .font(.body)
                    .foregroundStyle(color)
                    .frame(width: 26)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundStyle(.primary)
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)

            if !isLast {
                Divider()
                    .padding(.leading, 56)
            }
        }
    }
}

struct SectionHeader: View {
    let title: String

    var body: some View {
        Text(title)
            .font(.footnote)
            .fontWeight(.semibold)
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
            .padding(.horizontal, 4)
            .padding(.bottom, 8)
    }
}

#Preview {
    HomeView(selectedTab: .constant(.home))
        .environmentObject(HistoryStore())
}
