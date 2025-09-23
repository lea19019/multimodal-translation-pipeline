import React from 'react';
import { BarChart3, TrendingUp, Clock, Zap, Target, Cpu } from 'lucide-react';

const MetricsView: React.FC = () => {
  // Mock metrics data
  const performanceMetrics = {
    bleu: {
      current: 42.3,
      previous: 38.7,
      trend: 'up'
    },
    comet: {
      current: 0.847,
      previous: 0.823,
      trend: 'up'
    },
    latency: {
      current: 2.14,
      previous: 2.89,
      trend: 'down'
    },
    throughput: {
      current: 18.7,
      previous: 15.2,
      trend: 'up'
    }
  };

  const recentTranslations = [
    { id: 1, type: 'text-to-text', sourceLang: 'en', targetLang: 'es', bleu: 45.2, latency: 1.8, timestamp: '2 minutes ago' },
    { id: 2, type: 'speech-to-speech', sourceLang: 'fr', targetLang: 'de', bleu: 38.9, latency: 3.2, timestamp: '5 minutes ago' },
    { id: 3, type: 'text-to-speech', sourceLang: 'ja', targetLang: 'en', mos: 4.2, latency: 2.7, timestamp: '8 minutes ago' },
    { id: 4, type: 'speech-to-text', sourceLang: 'es', targetLang: 'en', wer: 12.5, latency: 1.5, timestamp: '12 minutes ago' },
  ];

  const getTrendIcon = (trend: string) => {
    return trend === 'up' ? (
      <TrendingUp className="h-4 w-4 text-green-400" />
    ) : (
      <TrendingUp className="h-4 w-4 text-red-400 rotate-180" />
    );
  };

  const getTrendColor = (trend: string) => {
    return trend === 'up' ? 'text-green-400' : 'text-red-400';
  };

  const getTranslationTypeIcon = (type: string) => {
    switch (type) {
      case 'text-to-text': return '📝';
      case 'text-to-speech': return '🔊';
      case 'speech-to-text': return '🎤';
      case 'speech-to-speech': return '🗣️';
      default: return '💬';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white">Analytics Dashboard</h2>
        <p className="text-slate-400 mt-1">
          Monitor translation pipeline performance and metrics
        </p>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card">
          <div className="card-content">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-400">BLEU Score</p>
                <p className="text-2xl font-bold text-blue-400">{performanceMetrics.bleu.current}</p>
                <div className="flex items-center space-x-1 mt-2">
                  {getTrendIcon(performanceMetrics.bleu.trend)}
                  <span className={`text-sm ${getTrendColor(performanceMetrics.bleu.trend)}`}>
                    +{(performanceMetrics.bleu.current - performanceMetrics.bleu.previous).toFixed(1)}
                  </span>
                </div>
              </div>
              <Target className="h-8 w-8 text-blue-400 opacity-60" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-content">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-400">COMET Score</p>
                <p className="text-2xl font-bold text-green-400">{performanceMetrics.comet.current}</p>
                <div className="flex items-center space-x-1 mt-2">
                  {getTrendIcon(performanceMetrics.comet.trend)}
                  <span className={`text-sm ${getTrendColor(performanceMetrics.comet.trend)}`}>
                    +{(performanceMetrics.comet.current - performanceMetrics.comet.previous).toFixed(3)}
                  </span>
                </div>
              </div>
              <BarChart3 className="h-8 w-8 text-green-400 opacity-60" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-content">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-400">Avg Latency</p>
                <p className="text-2xl font-bold text-yellow-400">{performanceMetrics.latency.current}s</p>
                <div className="flex items-center space-x-1 mt-2">
                  {getTrendIcon(performanceMetrics.latency.trend)}
                  <span className={`text-sm ${getTrendColor(performanceMetrics.latency.trend)}`}>
                    -{(performanceMetrics.latency.previous - performanceMetrics.latency.current).toFixed(2)}s
                  </span>
                </div>
              </div>
              <Clock className="h-8 w-8 text-yellow-400 opacity-60" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-content">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-400">Throughput</p>
                <p className="text-2xl font-bold text-purple-400">{performanceMetrics.throughput.current}/min</p>
                <div className="flex items-center space-x-1 mt-2">
                  {getTrendIcon(performanceMetrics.throughput.trend)}
                  <span className={`text-sm ${getTrendColor(performanceMetrics.throughput.trend)}`}>
                    +{(performanceMetrics.throughput.current - performanceMetrics.throughput.previous).toFixed(1)}
                  </span>
                </div>
              </div>
              <Zap className="h-8 w-8 text-purple-400 opacity-60" />
            </div>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold text-white">Performance Trends</h3>
          </div>
          <div className="card-content">
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-dark-900 rounded-md">
                <div>
                  <div className="text-sm font-medium text-slate-200">BLEU Score Trend</div>
                  <div className="text-xs text-slate-400">Last 7 days</div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-blue-400">↗ +9.3%</div>
                  <div className="text-xs text-green-400">Improving</div>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-4 bg-dark-900 rounded-md">
                <div>
                  <div className="text-sm font-medium text-slate-200">Translation Latency</div>
                  <div className="text-xs text-slate-400">Average response time</div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-green-400">↘ -26%</div>
                  <div className="text-xs text-green-400">Faster</div>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-4 bg-dark-900 rounded-md">
                <div>
                  <div className="text-sm font-medium text-slate-200">Model Accuracy</div>
                  <div className="text-xs text-slate-400">Cross-validation score</div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-blue-400">↗ +5.7%</div>
                  <div className="text-xs text-green-400">Better</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold text-white">System Resources</h3>
          </div>
          <div className="card-content">
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-slate-300">CPU Usage</span>
                  <span className="text-slate-400">67%</span>
                </div>
                <div className="w-full bg-dark-800 rounded-full h-2">
                  <div className="bg-blue-500 h-2 rounded-full" style={{ width: '67%' }}></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-slate-300">GPU Memory</span>
                  <span className="text-slate-400">82%</span>
                </div>
                <div className="w-full bg-dark-800 rounded-full h-2">
                  <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '82%' }}></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-slate-300">RAM Usage</span>
                  <span className="text-slate-400">45%</span>
                </div>
                <div className="w-full bg-dark-800 rounded-full h-2">
                  <div className="bg-green-500 h-2 rounded-full" style={{ width: '45%' }}></div>
                </div>
              </div>
              
              <div className="pt-4 border-t border-dark-700">
                <div className="flex items-center space-x-2 text-sm text-slate-400">
                  <Cpu className="h-4 w-4" />
                  <span>3 active model instances</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Translations */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold text-white">Recent Translations</h3>
        </div>
        <div className="card-content">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-dark-700">
                  <th className="text-left py-3 text-slate-300">Type</th>
                  <th className="text-left py-3 text-slate-300">Languages</th>
                  <th className="text-left py-3 text-slate-300">Quality</th>
                  <th className="text-left py-3 text-slate-300">Latency</th>
                  <th className="text-left py-3 text-slate-300">Time</th>
                </tr>
              </thead>
              <tbody>
                {recentTranslations.map((translation) => (
                  <tr key={translation.id} className="border-b border-dark-800 hover:bg-dark-900/50">
                    <td className="py-3">
                      <div className="flex items-center space-x-2">
                        <span className="text-lg">{getTranslationTypeIcon(translation.type)}</span>
                        <span className="text-slate-300">{translation.type.replace('-', ' → ')}</span>
                      </div>
                    </td>
                    <td className="py-3 text-slate-300">
                      {translation.sourceLang.toUpperCase()} → {translation.targetLang.toUpperCase()}
                    </td>
                    <td className="py-3">
                      {translation.bleu && (
                        <span className="text-blue-400">BLEU: {translation.bleu}</span>
                      )}
                      {translation.mos && (
                        <span className="text-green-400">MOS: {translation.mos}</span>
                      )}
                      {translation.wer && (
                        <span className="text-yellow-400">WER: {translation.wer}%</span>
                      )}
                    </td>
                    <td className="py-3 text-slate-300">{translation.latency}s</td>
                    <td className="py-3 text-slate-400">{translation.timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Model Performance Comparison */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold text-white">Pipeline Performance Comparison</h3>
        </div>
        <div className="card-content">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-dark-900 rounded-lg p-4">
              <h4 className="text-sm font-medium text-slate-200 mb-3">Baseline Pipeline</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">BLEU:</span>
                  <span className="text-blue-400">38.2</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Latency:</span>
                  <span className="text-yellow-400">3.1s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Throughput:</span>
                  <span className="text-purple-400">12/min</span>
                </div>
              </div>
            </div>
            
            <div className="bg-dark-900 rounded-lg p-4 ring-2 ring-blue-500">
              <h4 className="text-sm font-medium text-slate-200 mb-3">Advanced Pipeline</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">BLEU:</span>
                  <span className="text-blue-400">42.3</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Latency:</span>
                  <span className="text-yellow-400">2.1s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Throughput:</span>
                  <span className="text-purple-400">19/min</span>
                </div>
              </div>
            </div>
            
            <div className="bg-dark-900 rounded-lg p-4">
              <h4 className="text-sm font-medium text-slate-200 mb-3">Experimental Pipeline</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">BLEU:</span>
                  <span className="text-blue-400">44.7</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Latency:</span>
                  <span className="text-yellow-400">4.2s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Throughput:</span>
                  <span className="text-purple-400">8/min</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricsView;