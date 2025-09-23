import React from 'react';
import { Settings, Cpu, Mic, Volume2, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import { PipelineConfig, ModelInfo } from '../../shared/types';

interface PipelineManagerProps {
  pipelines: PipelineConfig[];
  selectedPipeline: PipelineConfig | null;
  onPipelineSelect: (pipeline: PipelineConfig) => void;
  onPipelinesUpdate: (pipelines: PipelineConfig[]) => void;
}

const PipelineManager: React.FC<PipelineManagerProps> = ({
  pipelines,
  selectedPipeline,
  onPipelineSelect
}) => {
  const getModelIcon = (type: string) => {
    switch (type) {
      case 'asr': return Mic;
      case 'nmt': return FileText;
      case 'tts': return Volume2;
      default: return Cpu;
    }
  };

  const getModelStatusColor = (model: ModelInfo) => {
    // Mock status logic - in real app, this would check model availability
    return 'text-green-400';
  };

  const renderModelCard = (model: ModelInfo | undefined, type: string) => {
    const Icon = getModelIcon(type);
    
    if (!model) {
      return (
        <div className="bg-dark-900 border border-dark-600 rounded-md p-4 opacity-50">
          <div className="flex items-center space-x-3">
            <Icon className="h-5 w-5 text-slate-500" />
            <div>
              <div className="text-sm text-slate-500">No {type.toUpperCase()} model</div>
              <div className="text-xs text-slate-600">Not configured</div>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="bg-dark-900 border border-dark-600 rounded-md p-4 hover:border-dark-500 transition-colors">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Icon className="h-5 w-5 text-blue-400" />
            <div>
              <div className="text-sm font-medium text-slate-200">{model.name}</div>
              <div className="text-xs text-slate-400">Version {model.version}</div>
            </div>
          </div>
          <CheckCircle className={`h-4 w-4 ${getModelStatusColor(model)}`} />
        </div>
        {model.description && (
          <div className="mt-2 text-xs text-slate-500">{model.description}</div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white">Pipeline Management</h2>
        <p className="text-slate-400 mt-1">
          Configure and manage your translation pipeline architectures
        </p>
      </div>

      {/* Pipeline Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {pipelines.map((pipeline) => (
          <div
            key={pipeline.id}
            className={`card cursor-pointer transition-all duration-200 hover:shadow-xl ${
              selectedPipeline?.id === pipeline.id
                ? 'ring-2 ring-blue-500 border-blue-500'
                : 'hover:border-dark-600'
            }`}
            onClick={() => onPipelineSelect(pipeline)}
          >
            <div className="card-header">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-white">{pipeline.name}</h3>
                  <p className="text-sm text-slate-400">{pipeline.description}</p>
                </div>
                <Settings className="h-5 w-5 text-slate-400" />
              </div>
            </div>
            
            <div className="card-content">
              <div className="space-y-3">
                <div>
                  <div className="text-xs font-medium text-slate-300 mb-2">Model Architecture</div>
                  <div className="space-y-2">
                    {renderModelCard(pipeline.models.asr, 'asr')}
                    {renderModelCard(pipeline.models.nmt, 'nmt')}
                    {renderModelCard(pipeline.models.tts, 'tts')}
                  </div>
                </div>
                
                <div className="pt-3 border-t border-dark-700">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400">Pipeline Status</span>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span className="text-green-400">Ready</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
        
        {/* Add New Pipeline Card */}
        <div className="card border-dashed border-dark-600 hover:border-dark-500 transition-colors cursor-pointer">
          <div className="card-content flex flex-col items-center justify-center py-12 text-center">
            <div className="w-12 h-12 bg-dark-800 rounded-full flex items-center justify-center mb-4">
              <Settings className="h-6 w-6 text-slate-400" />
            </div>
            <h3 className="text-lg font-medium text-slate-300 mb-2">Add New Pipeline</h3>
            <p className="text-sm text-slate-500">
              Create a custom pipeline configuration
            </p>
          </div>
        </div>
      </div>

      {/* Selected Pipeline Details */}
      {selectedPipeline && (
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold text-white">
              Pipeline Details: {selectedPipeline.name}
            </h3>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* ASR Model */}
              {selectedPipeline.models.asr && (
                <div>
                  <h4 className="text-sm font-medium text-slate-200 mb-3 flex items-center space-x-2">
                    <Mic className="h-4 w-4" />
                    <span>Automatic Speech Recognition</span>
                  </h4>
                  <div className="bg-dark-900 border border-dark-600 rounded-md p-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-xs text-slate-400">Model:</span>
                        <span className="text-xs text-slate-200">{selectedPipeline.models.asr.name}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-slate-400">Version:</span>
                        <span className="text-xs text-slate-200">{selectedPipeline.models.asr.version}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-slate-400">Status:</span>
                        <span className="text-xs text-green-400">Active</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* NMT Model */}
              {selectedPipeline.models.nmt && (
                <div>
                  <h4 className="text-sm font-medium text-slate-200 mb-3 flex items-center space-x-2">
                    <FileText className="h-4 w-4" />
                    <span>Neural Machine Translation</span>
                  </h4>
                  <div className="bg-dark-900 border border-dark-600 rounded-md p-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-xs text-slate-400">Model:</span>
                        <span className="text-xs text-slate-200">{selectedPipeline.models.nmt.name}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-slate-400">Version:</span>
                        <span className="text-xs text-slate-200">{selectedPipeline.models.nmt.version}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-slate-400">Status:</span>
                        <span className="text-xs text-green-400">Active</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* TTS Model */}
              {selectedPipeline.models.tts && (
                <div>
                  <h4 className="text-sm font-medium text-slate-200 mb-3 flex items-center space-x-2">
                    <Volume2 className="h-4 w-4" />
                    <span>Text-to-Speech</span>
                  </h4>
                  <div className="bg-dark-900 border border-dark-600 rounded-md p-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-xs text-slate-400">Model:</span>
                        <span className="text-xs text-slate-200">{selectedPipeline.models.tts.name}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-slate-400">Version:</span>
                        <span className="text-xs text-slate-200">{selectedPipeline.models.tts.version}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-xs text-slate-400">Status:</span>
                        <span className="text-xs text-green-400">Active</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-6 pt-6 border-t border-dark-700">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-medium text-slate-200">Performance Capabilities</h4>
                  <p className="text-xs text-slate-400">Estimated throughput and latency</p>
                </div>
                <div className="flex items-center space-x-4 text-xs">
                  <div className="text-center">
                    <div className="font-medium text-blue-400">~2.5s</div>
                    <div className="text-slate-500">Avg Latency</div>
                  </div>
                  <div className="text-center">
                    <div className="font-medium text-green-400">~15/min</div>
                    <div className="text-slate-500">Throughput</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Pipeline Performance Comparison */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold text-white">Pipeline Comparison</h3>
        </div>
        <div className="card-content">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-dark-700">
                  <th className="text-left py-3 text-slate-300">Pipeline</th>
                  <th className="text-left py-3 text-slate-300">ASR Model</th>
                  <th className="text-left py-3 text-slate-300">NMT Model</th>
                  <th className="text-left py-3 text-slate-300">TTS Model</th>
                  <th className="text-left py-3 text-slate-300">Status</th>
                </tr>
              </thead>
              <tbody>
                {pipelines.map((pipeline) => (
                  <tr 
                    key={pipeline.id} 
                    className={`border-b border-dark-800 hover:bg-dark-900/50 cursor-pointer ${
                      selectedPipeline?.id === pipeline.id ? 'bg-blue-500/10' : ''
                    }`}
                    onClick={() => onPipelineSelect(pipeline)}
                  >
                    <td className="py-3">
                      <div>
                        <div className="font-medium text-slate-200">{pipeline.name}</div>
                        <div className="text-xs text-slate-400">{pipeline.description}</div>
                      </div>
                    </td>
                    <td className="py-3 text-slate-300">
                      {pipeline.models.asr?.name || 'None'}
                    </td>
                    <td className="py-3 text-slate-300">
                      {pipeline.models.nmt?.name || 'None'}
                    </td>
                    <td className="py-3 text-slate-300">
                      {pipeline.models.tts?.name || 'None'}
                    </td>
                    <td className="py-3">
                      <div className="flex items-center space-x-1">
                        <CheckCircle className="h-3 w-3 text-green-400" />
                        <span className="text-green-400 text-xs">Ready</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PipelineManager;