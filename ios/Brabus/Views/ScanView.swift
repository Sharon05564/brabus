// ScanView.swift
// Brabus — On-Device Document Intelligence Assistant
//
// Provides two ways to get text into the app:
//   1. Photo picker → Vision OCR → extracted text shown for review
//   2. Direct text paste (fallback / always available)
//
// All OCR processing happens on-device using Apple's Vision framework.
// No image or text is sent to any server.

import SwiftUI
import PhotosUI
import Vision

struct ScanView: View {

    @Binding var selectedTab: ContentView.Tab

    // Photo picker state
    @State private var pickerItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?

    // OCR state
    @State private var isProcessing = false
    @State private var ocrText = ""
    @State private var ocrError: String?

    // Sheet for reviewing extracted text before analysis
    @State private var showReviewSheet = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    instructionCard
                    photoPickerCard
                    pasteCard
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 24)
            }
            .navigationTitle("Scan Document")
            .navigationBarTitleDisplayMode(.large)
            .background(Color(.systemGroupedBackground))
            .sheet(isPresented: $showReviewSheet) {
                OCRReviewSheet(
                    extractedText: $ocrText,
                    onAnalyze: {
                        showReviewSheet = false
                        selectedTab = .analyze
                    }
                )
            }
            .onChange(of: pickerItem) { _, newItem in
                guard let newItem else { return }
                Task { await processPickerItem(newItem) }
            }
        }
    }

    // MARK: - Subviews

    private var instructionCard: some View {
        HStack(spacing: 14) {
            Image(systemName: "viewfinder.circle.fill")
                .font(.title2)
                .foregroundStyle(.indigo)
            VStack(alignment: .leading, spacing: 3) {
                Text("Scan a Document")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Text("Select a photo to extract text using Vision OCR, or paste text directly.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var photoPickerCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader(title: "From Photo Library")

            PhotosPicker(
                selection: $pickerItem,
                matching: .images,
                photoLibrary: .shared()
            ) {
                HStack {
                    Image(systemName: "photo.on.rectangle.angled")
                        .font(.title3)
                    Text("Choose Photo")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(Color.indigo)
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .buttonStyle(.plain)

            if isProcessing {
                HStack(spacing: 10) {
                    ProgressView()
                        .tint(.indigo)
                    Text("Extracting text with Vision OCR…")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }

            if let error = ocrError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }

            if let image = selectedImage {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Selected Image")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 180)
                        .clipShape(RoundedRectangle(cornerRadius: 10))

                    if !ocrText.isEmpty {
                        Button {
                            showReviewSheet = true
                        } label: {
                            Label("Review & Analyze Extracted Text", systemImage: "arrow.right.circle.fill")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 13)
                                .background(Color.indigo.opacity(0.12))
                                .foregroundStyle(.indigo)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var pasteCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader(title: "Or Paste Text Directly")

            NavigationLink {
                AnalyzeView()
            } label: {
                HStack {
                    Image(systemName: "text.cursor")
                        .font(.title3)
                    Text("Go to Analyze Tab")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(Color(.tertiarySystemGroupedBackground))
                .foregroundStyle(.indigo)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .strokeBorder(Color.indigo.opacity(0.3), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)

            Text("Type or paste document text in the Analyze tab and tap \"Analyze Document\" to classify it on-device.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - OCR Processing

    @MainActor
    private func processPickerItem(_ item: PhotosPickerItem) async {
        isProcessing = true
        ocrError = nil
        ocrText = ""
        selectedImage = nil

        do {
            guard let data = try await item.loadTransferable(type: Data.self),
                  let uiImage = UIImage(data: data) else {
                ocrError = "Could not load the selected image."
                isProcessing = false
                return
            }
            selectedImage = uiImage

            let extracted = try await performOCR(on: uiImage)
            ocrText = extracted
            if ocrText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                ocrError = "No text was detected in this image. Try a clearer photo."
            } else {
                showReviewSheet = true
            }
        } catch {
            ocrError = "OCR failed: \(error.localizedDescription)"
        }
        isProcessing = false
    }

    /// Runs VNRecognizeTextRequest on the provided UIImage and returns
    /// all recognised strings joined by newlines.
    private func performOCR(on image: UIImage) async throws -> String {
        guard let cgImage = image.cgImage else {
            throw OCRError.invalidImage
        }

        return try await withCheckedThrowingContinuation { continuation in
            let request = VNRecognizeTextRequest { request, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                let observations = request.results as? [VNRecognizedTextObservation] ?? []
                let lines = observations.compactMap {
                    $0.topCandidates(1).first?.string
                }
                continuation.resume(returning: lines.joined(separator: "\n"))
            }
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    enum OCRError: LocalizedError {
        case invalidImage
        var errorDescription: String? { "Could not process the image for OCR." }
    }
}

// MARK: - OCR Review Sheet

private struct OCRReviewSheet: View {
    @Binding var extractedText: String
    let onAnalyze: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Review Extracted Text")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 20)
                        .padding(.top, 16)

                    TextEditor(text: $extractedText)
                        .font(.system(.body, design: .monospaced))
                        .frame(minHeight: 260)
                        .padding(12)
                        .background(Color(.secondarySystemGroupedBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .padding(.horizontal, 16)
                }

                Spacer()

                VStack(spacing: 10) {
                    Text("Edit the text above if OCR made mistakes, then tap Analyze.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 20)

                    Button {
                        onAnalyze()
                    } label: {
                        Label("Analyze Document", systemImage: "brain.head.profile")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(Color.indigo)
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                            .padding(.horizontal, 20)
                    }
                    .buttonStyle(.plain)
                    .disabled(extractedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                .padding(.bottom, 32)
            }
            .navigationTitle("Extracted Text")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                }
            }
            .background(Color(.systemGroupedBackground))
        }
    }
}

#Preview {
    ScanView(selectedTab: .constant(.scan))
        .environmentObject(HistoryStore())
}
