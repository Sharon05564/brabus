// BrabusApp.swift
// Brabus — On-Device Document Intelligence Assistant
//
// Main entry point for the iOS app.
// All ML inference runs on-device via Core ML; no network calls are made.

import SwiftUI

@main
struct BrabusApp: App {

    /// Shared history store injected into the environment so every view can
    /// read and append analysis results without passing them down manually.
    @StateObject private var historyStore = HistoryStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(historyStore)
        }
    }
}

// MARK: - History Store

/// A lightweight in-memory store for the current session's analysis results.
/// This is intentionally session-only; results are cleared when the app relaunches.
/// A real production app could persist results with SwiftData or CoreData.
final class HistoryStore: ObservableObject {
    @Published private(set) var results: [DocumentAnalysisResult] = []

    /// Text queued from ScanView's OCR pipeline for AnalyzeView to pick up
    /// and run automatically. Cleared by AnalyzeView after consuming.
    @Published var pendingText: String = ""

    func append(_ result: DocumentAnalysisResult) {
        results.insert(result, at: 0)  // newest first
    }

    func clear() {
        results.removeAll()
    }
}
