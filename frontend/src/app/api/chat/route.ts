import { DEFAULT_AGENT_ID, jobbuddyApiUrl, jobbuddyHeaders } from '@/lib/jobbuddyApi';

// The Node runtime, because the edge runtime buffers differently and this route's
// whole job is to not buffer. force-dynamic keeps Next from trying to cache a stream.
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const logApiCall = (phase: string, data: Record<string, unknown>, level: 'INFO' | 'WARN' | 'ERROR' = 'INFO') => {
  const timestamp = new Date().toISOString();
  const logLevel = process.env.NODE_ENV === 'development' ? 'DEBUG' : 'INFO';
  
  // In production, only log ERROR and WARN levels unless DEBUG is explicitly enabled
  if (process.env.NODE_ENV === 'production' && level === 'INFO' && process.env.ENABLE_DEBUG_LOGS !== 'true') {
    return;
  }
  
  const logPrefix = level === 'ERROR' ? '🔴' : level === 'WARN' ? '🟡' : '🔵';
  console.log(`\n${logPrefix} === API CALL LOG [${level}] [${phase}] - ${timestamp} ===`);
  console.log(JSON.stringify(data, null, 2));
  console.log('=====================================\n');
};

export async function POST(req: Request) {
  const requestStartTime = Date.now();
  
  try {
    const { messages, userId, threadId, mode = 'stream', agentId = DEFAULT_AGENT_ID } = await req.json();
    const latestMessage = messages[messages.length - 1]?.content || '';
    
    // Detailed request start log
    logApiCall('REQUEST_START', {
      timestamp: new Date().toISOString(),
      requestId: `req_${requestStartTime}`,
      requestHeaders: Object.fromEntries(req.headers.entries()),
      requestUrl: req.url,
      requestMethod: req.method,
      input: {
        userId: userId,
        threadId: threadId,
        agentId: agentId,
        messageCount: messages.length,
        latestMessage: latestMessage,
        allMessages: messages
      },
      environment: {
        nodeEnv: process.env.NODE_ENV,
        apiEnv: process.env.NEXT_PUBLIC_API_ENV,
        hasAuthToken: !!process.env.JOBBUDDY_API_TOKEN
      }
    });

    // Validate and ensure userId is an integer
    const validateUserId = (id: unknown): number => {
      if (id === null || id === undefined) {
        throw new Error('user_id is required and must be provided');
      }
      const numericId = Number(id);
      if (!Number.isInteger(numericId) || numericId <= 0) {
        throw new Error('user_id must be a positive integer');
      }
      return numericId;
    };
    
    const finalUserId = validateUserId(userId);
    
    interface RequestBody {
      message: string;
      user_id: number;
      thread_id?: string;
      stream_tokens?: boolean;
    }
    
    const requestBody: RequestBody = {
      message: latestMessage,
      user_id: finalUserId
    };
    
    // Only stream mode needs stream_tokens
    if (mode === 'stream') {
      requestBody.stream_tokens = true;
    }
    
    if (threadId) {
      requestBody.thread_id = threadId;
    }

    const isLocal = process.env.NEXT_PUBLIC_API_ENV === 'local';
    let apiUrl: string | null = null;
    try {
      apiUrl = jobbuddyApiUrl();
    } catch {
      apiUrl = null;
    }

    // Check if API URL is configured
    if (!apiUrl) {
      const errorMsg =
        'Backend API URL not configured. Set JOBBUDDY_API_URL (see .env.example).';
      
      logApiCall('CONFIGURATION_ERROR', {
        timestamp: new Date().toISOString(),
        requestId: `req_${requestStartTime}`,
        error: {
          message: errorMsg,
          isLocal: isLocal,
          apiEnv: process.env.NEXT_PUBLIC_API_ENV
        }
      }, 'ERROR');
      
      return new Response(JSON.stringify({
        content: `Configuration Error: ${errorMsg}`,
        threadId: null,
        userId: null
      }), { 
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    
    const endpoint = mode === 'stream' ? 'stream' : 'invoke';
    const fullApiUrl = `${apiUrl}/${agentId}/${endpoint}`;
    
    // Detailed external API request log
    logApiCall('EXTERNAL_API_REQUEST', {
      timestamp: new Date().toISOString(),
      requestId: `req_${requestStartTime}`,
      externalApi: {
        url: fullApiUrl,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          hasAuthToken: !!process.env.JOBBUDDY_API_TOKEN
        },
        body: requestBody
      },
      processedData: {
        originalUserId: userId,
        finalUserId: finalUserId,
        originalThreadId: threadId,
        messageLength: latestMessage.length,
        isLocal: isLocal
      }
    });
    
    const response = await fetch(fullApiUrl, {
      method: 'POST',
      headers: jobbuddyHeaders(),
      body: JSON.stringify(requestBody),
      cache: 'no-store',
    });

    // Detailed external API response log
    const responseTime = Date.now() - requestStartTime;
    logApiCall('EXTERNAL_API_RESPONSE', {
      timestamp: new Date().toISOString(),
      requestId: `req_${requestStartTime}`,
      response: {
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers.entries()),
        ok: response.ok
      },
      performance: {
        responseTime: `${responseTime}ms`,
        startTime: requestStartTime
      }
    });

    if (!response.ok) {
      const errorText = await response.text();
      
      logApiCall('API_ERROR', {
        timestamp: new Date().toISOString(),
        requestId: `req_${requestStartTime}`,
        error: {
          status: response.status,
          statusText: response.statusText,
          errorText: errorText,
          responseTime: `${responseTime}ms`
        }
      }, 'ERROR');
      
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Handle invoke mode synchronous response
    if (mode === 'invoke') {
      const invokeResponse = await response.json();
      
      logApiCall('INVOKE_SUCCESS', {
        timestamp: new Date().toISOString(),
        requestId: `req_${requestStartTime}`,
        response: {
          contentLength: invokeResponse.output?.content?.length || 0,
          threadId: invokeResponse.thread_id,
          userId: invokeResponse.user_id
        }
      });
      
      return Response.json({
        content: invokeResponse.output?.content || '',
        threadId: invokeResponse.thread_id,
        userId: invokeResponse.user_id,
        section: invokeResponse.output?.custom_data?.section || null,
        mode: 'invoke'
      });
    }

    // Return streaming response
    const encoder = new TextEncoder();
    let finalContent = '';
    let finalThreadId: string | null = null;
    let finalUserId_response: number | null = null;
    let finalSection: Record<string, unknown> | null = null;
    let chunkCount = 0;
    let tokenCount = 0;
    let lastActivityTime = Date.now();

    const stream = new ReadableStream({
      async start(controller) {
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        const STREAM_TIMEOUT = 30000; // 30 second timeout
        let timeoutId: NodeJS.Timeout | null = null;
        let isClosed = false; // Flag to prevent duplicate close

        // Safely write data
        const safeEnqueue = (data: Uint8Array) => {
          if (!isClosed) {
            try {
              controller.enqueue(data);
            } catch (e) {
              console.warn('Failed to enqueue data:', e);
            }
          }
        };

        // Safely close stream
        const safeClose = () => {
          if (!isClosed) {
            isClosed = true;
            if (timeoutId) clearTimeout(timeoutId);
            try {
              controller.close();
            } catch (e) {
              console.warn('Failed to close controller:', e);
            }
          }
        };

        // Set stream timeout monitoring
        const resetTimeout = () => {
          if (timeoutId) clearTimeout(timeoutId);
          if (!isClosed) {
            timeoutId = setTimeout(() => {
              logApiCall('STREAM_TIMEOUT', {
                timestamp: new Date().toISOString(),
                requestId: `req_${requestStartTime}`,
                timeoutInfo: {
                  timeoutDuration: STREAM_TIMEOUT,
                  lastActivityTime: new Date(lastActivityTime).toISOString(),
                  chunkCount: chunkCount,
                  tokenCount: tokenCount,
                  inactiveTime: `${Date.now() - lastActivityTime}ms`
                }
              }, 'ERROR');
              if (!isClosed) {
                controller.error(new Error('Stream timeout - no data received for 30 seconds'));
                isClosed = true;
              }
            }, STREAM_TIMEOUT);
          }
        };
        
        resetTimeout();
        
        logApiCall('STREAM_READER_INIT', {
          timestamp: new Date().toISOString(),
          requestId: `req_${requestStartTime}`,
          streamSetup: {
            hasReader: !!reader,
            responseBodyExists: !!response.body,
            timeoutConfigured: STREAM_TIMEOUT
          }
        });
        
        if (!reader) {
          logApiCall('STREAM_ERROR', {
            timestamp: new Date().toISOString(),
            requestId: `req_${requestStartTime}`,
            error: {
              type: 'NO_READER',
              message: 'No reader available from response body'
            }
          }, 'ERROR');
          safeClose();
          return;
        }

        try {
          while (true) {
            const readStart = Date.now();
            const { done, value } = await reader.read();
            const readTime = Date.now() - readStart;
            
            if (done) {
              logApiCall('STREAM_COMPLETE', {
                timestamp: new Date().toISOString(),
                requestId: `req_${requestStartTime}`,
                streamStats: {
                  totalChunks: chunkCount,
                  totalTokens: tokenCount,
                  finalContentLength: finalContent.length,
                  totalStreamTime: `${Date.now() - requestStartTime}ms`,
                  lastActivityTime: new Date(lastActivityTime).toISOString()
                }
              });
              break;
            }

            chunkCount++;
            lastActivityTime = Date.now();
            resetTimeout(); // Reset timeout timer
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            logApiCall('STREAM_CHUNK_RECEIVED', {
              timestamp: new Date().toISOString(),
              requestId: `req_${requestStartTime}`,
              chunkData: {
                chunkNumber: chunkCount,
                chunkSize: value?.length || 0,
                readTime: `${readTime}ms`,
                linesCount: lines.length,
                rawChunk: chunk.substring(0, 200) + (chunk.length > 200 ? '...' : ''),
                chunkLength: chunk.length
              }
            });

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') {
                  // Send final complete response data
                  const finalResponseData = {
                    type: 'final_response',
                    content: finalContent,
                    threadId: finalThreadId,
                    userId: finalUserId_response || finalUserId,
                    section: finalSection
                  };
                  
                  logApiCall('STREAM_SENDING_FINAL', {
                    timestamp: new Date().toISOString(),
                    requestId: `req_${requestStartTime}`,
                    finalData: {
                      contentLength: finalContent.length,
                      threadId: finalThreadId,
                      userId: finalUserId_response || finalUserId,
                      hasSection: !!finalSection,
                      totalTokens: tokenCount,
                      totalChunks: chunkCount
                    }
                  });
                  
                  safeEnqueue(encoder.encode(`data: ${JSON.stringify(finalResponseData)}\n\n`));
                  safeEnqueue(encoder.encode('data: [DONE]\n\n'));
                  safeClose();
                  return;
                }

                try {
                  const parsed = JSON.parse(data);
                  
                  logApiCall('STREAM_DATA_PARSED', {
                    timestamp: new Date().toISOString(),
                    requestId: `req_${requestStartTime}`,
                    parsedData: {
                      type: parsed.type,
                      contentLength: parsed.content?.length || 0,
                      hasRunId: !!parsed.content?.run_id,
                      hasCustomData: !!parsed.content?.custom_data,
                      rawDataSample: data.substring(0, 100) + (data.length > 100 ? '...' : '')
                    }
                  });
                  
                  // Handle different types of streaming data
                  if (parsed.type === 'token') {
                    tokenCount++;
                    finalContent += parsed.content;
                  } else if (parsed.type === 'message') {
                    finalContent = parsed.content.content || finalContent;
                    if (parsed.content.run_id) {
                      finalThreadId = parsed.content.run_id;
                    }
                    if (parsed.content.custom_data) {
                      finalSection = parsed.content.custom_data.section;
                      finalUserId_response = parsed.content.custom_data.user_id;
                    }
                  } else if (parsed.type === 'section') {
                    finalSection = parsed.content;
                  }
                  
                  // Forward streaming data
                  safeEnqueue(encoder.encode(`data: ${JSON.stringify(parsed)}\n\n`));
                } catch (e) {
                  logApiCall('STREAM_PARSE_ERROR', {
                    timestamp: new Date().toISOString(),
                    requestId: `req_${requestStartTime}`,
                    parseError: {
                      error: e instanceof Error ? e.message : 'Unknown parse error',
                      rawData: data,
                      chunkNumber: chunkCount,
                      lineNumber: lines.indexOf(line)
                    }
                  }, 'WARN');
                  console.error('Failed to parse stream data:', e, data);
                }
              }
            }
          }
        } catch (error) {
          logApiCall('STREAM_PROCESSING_ERROR', {
            timestamp: new Date().toISOString(),
            requestId: `req_${requestStartTime}`,
            streamError: {
              error: error instanceof Error ? error.message : 'Unknown stream error',
              stack: error instanceof Error ? error.stack : undefined,
              chunkCount: chunkCount,
              tokenCount: tokenCount,
              lastActivityTime: new Date(lastActivityTime).toISOString(),
              streamDuration: `${Date.now() - requestStartTime}ms`
            }
          }, 'ERROR');
          console.error('Stream processing error:', error);
          if (!isClosed) {
            controller.error(error);
            isClosed = true;
          }
          if (timeoutId) clearTimeout(timeoutId);
        }
      }
    });

    logApiCall('STREAM_START', {
      timestamp: new Date().toISOString(),
      requestId: `req_${requestStartTime}`,
      streamSetup: {
        responseHeaders: Object.fromEntries(response.headers.entries())
      }
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        // no-transform matters as much as no-cache: without it an intermediary may
        // coalesce chunks and the stream silently stops looking like a stream.
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    });

  } catch (error) {
    const errorResponseTime = Date.now() - requestStartTime;
    
    logApiCall('REQUEST_ERROR', {
      timestamp: new Date().toISOString(),
      requestId: `req_${requestStartTime}`,
      error: {
        message: error instanceof Error ? error.message : 'Unknown error',
        stack: error instanceof Error ? error.stack : undefined,
        type: error?.constructor?.name || 'Unknown'
      },
      performance: {
        errorResponseTime: `${errorResponseTime}ms`,
        requestStartTime: requestStartTime,
        errorTime: Date.now()
      }
    }, 'ERROR');
    
    return new Response(JSON.stringify({
      content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      threadId: null,
      userId: null
    }), { 
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}