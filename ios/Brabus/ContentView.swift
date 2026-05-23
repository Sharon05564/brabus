// ContentView.swift
// Brabus — On-Device Document Intelligence Assistant
//
// Root navigation container using a TabView with five tabs.

import SwiftUI

struct ContentView: View {

    @State private var selectedTab: Tab = .home

    enum Tab: Int {
        case home, scan, analyze, history, about
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            HomeView(selectedTab: $selectedTab)
                .tabItem {
                    Label("Home", systemImage: "house.fill")
                }
                .tag(Tab.home)

            ScanView(selectedTab: $selectedTab)
                .tabItem {
                    Label("Scan", systemImage: "viewfinder.circle.fill")
                }
                .tag(Tab.scan)

            AnalyzeView()
                .tabItem {
                    Label("Analyze", systemImage: "text.magnifyingglass")
                }
                .tag(Tab.analyze)

            HistoryView()
                .tabItem {
                    Label("History", systemImage: "clock.fill")
                }
                .tag(Tab.history)

            AboutView()
                .tabItem {
                    Label("About", systemImage: "info.circle.fill")
                }
                .tag(Tab.about)
        }
        .tint(.indigo)
    }
}

#Preview {
    ContentView()
        .environmentObject(HistoryStore())
}
