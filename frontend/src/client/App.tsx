import React, { useState, useEffect } from 'react';
import { Settings, Zap, BarChart3, FileText, Mic, Volume2, Languages } from 'lucide-react';
import TranslationInterface from './components/TranslationInterface';
import PipelineManager from './components/PipelineManager';
import MetricsView from './components/MetricsView';
import { PipelineConfig } from '../shared/types';

type ActiveTab = 'translate' | 'pipelines' | 'metrics';

interface AppState {
  activeTab: ActiveTab;
  selectedPipeline: PipelineConfig | null;
  pipelines: PipelineConfig[];
}

function App() {
  const [state, setState] = useState<AppState>({
    activeTab: 'translate',
    selectedPipeline: null,
    pipelines: []
  });

  // Load pipelines on mount
  useEffect(() => {
    loadPipelines();
  }, []);

  const loadPipelines = async () => {
    try {
      const response = await fetch('/api/pipelines');
      const data = await response.json();
      
      if (data.data) {
        setState(prev => ({
          ...prev,
          pipelines: data.data,
          selectedPipeline: data.data[0] || null
        }));
      }
    } catch (error) {
      console.error('Failed to load pipelines:', error);
    }
  };

  const updateState = (updates: Partial<AppState>) => {
    setState(prev => ({ ...prev, ...updates }));
  };

  const navigationItems = [
    {
      id: 'translate' as const,
      label: 'Translation',
      icon: Languages,
      description: 'Test translations and evaluate models'
    },
    {
      id: 'pipelines' as const,
      label: 'Pipelines',
      icon: Settings,
      description: 'Manage pipeline configurations'
    },
    {
      id: 'metrics' as const,
      label: 'Analytics',
      icon: BarChart3,
      description: 'View performance metrics'
    }
  ];

  const renderContent = () => {
    switch (state.activeTab) {
      case 'translate':
        return (
          <TranslationInterface
            selectedPipeline={state.selectedPipeline}
            pipelines={state.pipelines}
            onPipelineChange={(pipeline) => updateState({ selectedPipeline: pipeline })}
          />
        );
      case 'pipelines':
        return (
          <PipelineManager
            pipelines={state.pipelines}
            selectedPipeline={state.selectedPipeline}
            onPipelineSelect={(pipeline) => updateState({ selectedPipeline: pipeline })}
            onPipelinesUpdate={(pipelines) => updateState({ pipelines })}
          />
        );
      case 'metrics':
        return <MetricsView />;
      default:
        return null;
    }
  };

  const getTranslationTypeIcon = (type: string) => {
    switch (type) {
      case 'text': return FileText;
      case 'speech': return Mic;
      case 'audio': return Volume2;
      default: return Zap;
    }
  };

  return (
    <div className="min-h-screen bg-dark-950">
      {/* Header */}
      <header className="bg-dark-900 border-b border-dark-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Zap className="h-8 w-8 text-blue-500" />
                <h1 className="text-xl font-bold text-white">
                  Translation Pipeline
                </h1>
              </div>
              {state.selectedPipeline && (
                <div className="hidden md:flex items-center space-x-2 text-sm text-slate-400">
                  <span>•</span>
                  <span>{state.selectedPipeline.name}</span>
                </div>
              )}
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1 text-sm text-slate-400">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>API Connected</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-dark-900 border-b border-dark-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const isActive = state.activeTab === item.id;
              
              return (
                <button
                  key={item.id}
                  onClick={() => updateState({ activeTab: item.id })}
                  className={`flex items-center space-x-2 px-3 py-4 text-sm font-medium border-b-2 transition-colors ${
                    isActive
                      ? 'border-blue-500 text-blue-400'
                      : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-600'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderContent()}
      </main>

      {/* Footer */}
      <footer className="bg-dark-900 border-t border-dark-800 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="text-center text-sm text-slate-400">
            Multimodal Translation Pipeline Dashboard • Built with React, TypeScript & Express
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;