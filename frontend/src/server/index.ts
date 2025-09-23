import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { TranslationRequest, TranslationResponse, PipelineConfig, EvaluationMetrics, ApiResponse } from '../shared/types.js';
import TranslationService from './translation-service.js';

const app = express();
const port = 3003;

// Initialize translation service
const translationService = new TranslationService();

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Configure multer for file uploads
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 } // 100MB limit
});

// Mock translation function

// Mock translation function
function mockTranslate(request: TranslationRequest): TranslationResponse {
  const processingTime = Math.random() * 2000 + 500; // 500-2500ms
  
  let output: string;
  let evaluationMetrics: EvaluationMetrics | undefined;
  
  // Determine if input is file (base64) or text
  const isFileInput = typeof request.input === 'string' && request.input.startsWith('data:') || 
                      (typeof request.input === 'string' && request.input.length > 500 && !request.input.includes(' '));
  
  // Extract text content for processing
  let textContent: string;
  if (isFileInput) {
    // Mock extraction from file - in real implementation, you'd process the actual file
    textContent = `Content extracted from ${request.sourceLang} file`;
  } else {
    textContent = request.input as string;
  }
  
  // Generate mock output based on type
  switch (request.type) {
    case 'text-to-text':
      // Can accept both text input and text files
      if (isFileInput) {
        output = `[${request.targetLang.toUpperCase()}] Translated content from uploaded file: ${textContent}`;
      } else {
        output = `[${request.targetLang.toUpperCase()}] Translated: ${textContent}`;
      }
      break;
      
    case 'text-to-speech':
      // Can accept both text input and text files, always outputs audio
      const sourceText = isFileInput ? textContent : textContent;
      output = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+n1vWoiCT+f3PbCdygF';
      console.log(`🔊 TTS: Converting "${sourceText}" to ${request.targetLang} speech`);
      break;
      
    case 'speech-to-text':
      // Expects audio file input, outputs text
      if (isFileInput) {
        output = `Transcribed from ${request.sourceLang} audio: "Hello, this is a mock transcription of the uploaded audio file."`;
      } else {
        output = `Error: Speech-to-text requires audio file input, but received text: "${textContent}"`;
      }
      break;
      
    case 'speech-to-speech':
      // Expects audio file input, outputs audio file
      if (isFileInput) {
        output = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+n1vWoiCT+f3PbCdygF';
        console.log(`🎙️ Speech-to-Speech: ${request.sourceLang} → ${request.targetLang}`);
      } else {
        output = `Error: Speech-to-speech requires audio file input, but received text: "${textContent}"`;
      }
      break;
  }
  
  // Generate evaluation metrics if in evaluation mode
  if (request.mode === 'evaluation') {
    evaluationMetrics = {
      bleu: Math.random() * 40 + 20, // 20-60
      comet: Math.random() * 0.4 + 0.6, // 0.6-1.0
      wer: Math.random() * 0.2 + 0.05, // 0.05-0.25
      mos: Math.random() * 2 + 3, // 3-5
      latency: processingTime,
      throughput: Math.random() * 50 + 10 // 10-60 items/sec
    };
  }
  
  return {
    id: request.id,
    output,
    evaluationMetrics,
    processingTime,
    status: 'success'
  };
}

// Routes
app.get('/api/health', async (_req, res) => {
  try {
    const health = await translationService.healthCheck();
    res.json(health);
  } catch (error) {
    res.status(500).json({ status: 'error', message: 'Health check failed' });
  }
});

app.get('/api/pipelines', async (_req, res) => {
  try {
    const pipelines = await translationService.getPipelines();
    const response: ApiResponse<PipelineConfig[]> = {
      data: pipelines,
      message: 'Pipeline configurations retrieved successfully'
    };
    res.json(response);
  } catch (error) {
    console.error('Error fetching pipelines:', error);
    res.status(500).json({ error: 'Failed to fetch pipelines' });
  }
});

app.get('/api/pipelines/:id', async (req, res) => {
  try {
    const pipeline = await translationService.getPipeline(req.params.id);
    if (!pipeline) {
      return res.status(404).json({ error: 'Pipeline not found' });
    }
    
    const response: ApiResponse<PipelineConfig> = {
      data: pipeline,
      message: 'Pipeline configuration retrieved successfully'
    };
    res.json(response);
  } catch (error) {
    console.error('Error fetching pipeline:', error);
    res.status(500).json({ error: 'Failed to fetch pipeline' });
  }
});

app.post('/api/translate', upload.single('file'), async (req, res) => {
  try {
    console.log('📥 Received translation request');
    console.log('Body keys:', Object.keys(req.body));
    console.log('File:', req.file ? `${req.file.originalname} (${req.file.size} bytes)` : 'None');
    console.log('Raw request data:', req.body.request ? 'JSON string' : 'direct form data');
    
    // Parse request data
    let requestData: any = {};
    
    if (req.body.request) {
      requestData = JSON.parse(req.body.request);
    } else {
      // Handle direct form data
      requestData = {
        id: req.body.id || Date.now().toString(),
        type: req.body.type,
        mode: req.body.mode || 'free',
        sourceLang: req.body.sourceLang,
        targetLang: req.body.targetLang,
        input: req.body.input,
        pipelineId: req.body.pipelineId || 'baseline',
        reference: req.body.reference
      };
    }

    // Ensure pipelineId is set regardless of how the request came in
    if (!requestData.pipelineId) {
      requestData.pipelineId = 'baseline';
    }

    console.log('🔧 Processed request data:', {
      type: requestData.type,
      sourceLang: requestData.sourceLang,
      targetLang: requestData.targetLang,
      pipelineId: requestData.pipelineId
    });

    // Handle file upload
    if (req.file) {
      if (requestData.type?.includes('speech')) {
        requestData.input = { audioData: req.file.buffer };
      } else {
        requestData.input = { text: req.file.buffer.toString('utf-8') };
      }
    } else if (typeof requestData.input === 'string') {
      requestData.input = { text: requestData.input };
    }

    // Validate required fields
    if (!requestData.type || !requestData.sourceLang || !requestData.targetLang) {
      return res.status(400).json({
        error: 'Missing required fields: type, sourceLang, targetLang'
      });
    }

    console.log(`🔄 Processing ${requestData.type} translation: ${requestData.sourceLang} → ${requestData.targetLang}`);

    // Process translation using the service
    const result = await translationService.processTranslation(requestData);

    const response: ApiResponse<TranslationResponse> = {
      data: result,
      message: result.success ? 'Translation completed successfully' : 'Translation failed'
    };

    res.json(response);
  } catch (error) {
    console.error('Translation error:', error);
    const response: ApiResponse<null> = {
      error: 'Internal server error during translation'
    };
    res.status(500).json(response);
  }
});

app.post('/api/translate/batch', upload.array('files'), async (req, res) => {
  try {
    const requests: TranslationRequest[] = JSON.parse(req.body.requests || '[]');
    const batchId = req.body.batchId || `batch_${Date.now()}`;
    
    // Process all requests
    const results = await Promise.all(
      requests.map(async (request) => {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 500));
        return mockTranslate(request);
      })
    );
    
    // Calculate overall metrics
    const successCount = results.filter(r => r.status === 'success').length;
    const overallMetrics = {
      averageBleu: results.reduce((sum, r) => sum + (r.evaluationMetrics?.bleu || 0), 0) / results.length,
      averageComet: results.reduce((sum, r) => sum + (r.evaluationMetrics?.comet || 0), 0) / results.length,
      averageLatency: results.reduce((sum, r) => sum + (r.processingTime || 0), 0) / results.length,
      successRate: (successCount / results.length) * 100
    };
    
    const response: ApiResponse<any> = {
      data: {
        batchId,
        results,
        overallMetrics
      },
      message: 'Batch translation completed successfully'
    };
    
    res.json(response);
  } catch (error) {
    console.error('Batch translation error:', error);
    const response: ApiResponse<null> = {
      error: 'Internal server error during batch translation'
    };
    res.status(500).json(response);
  }
});

app.listen(port, () => {
  console.log(`🚀 Translation Pipeline API running on http://localhost:${port}`);
  console.log(`📊 Health check: http://localhost:${port}/api/health`);
});

export default app;