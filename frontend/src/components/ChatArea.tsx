'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import JobBuddyWelcome from '@/components/JobBuddyWelcome';
import type { ResumeView } from '@/utils/resume';
import ProgressiveText from '@/components/ProgressiveText';
import {
  chatScrollBehavior,
  isNearChatBottom,
  type ChatScrollTrigger,
} from '@/utils/chatScroll';
import { planMessageEvent } from '@/utils/chatStream';


export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

interface Section {
  database_id: number;
  name: string;
  status: string;
}

/** One section, exactly as the backend's public projection sends it. */
export interface PublicSection {
  id: string;
  name: string;
  status: 'pending' | 'in_progress' | 'done';
}

/** The PR6 `completion` SSE event. Three fields, nothing internal. */
export interface CompletionState {
  collection_complete: boolean;
  artifact_available: boolean;
  sections: PublicSection[];
}

interface ChatAreaProps {
  selectedAgent: string;
  userId: number;
  mode: 'invoke' | 'stream';
  threadId: string | null;
  loadedMessages?: Message[];
  currentSection: Section | null;
  onThreadIdChange: (threadId: string) => void;
  onSectionUpdate: (section: Section) => void;
  onCompletionUpdate?: (completion: CompletionState) => void;
  onFirstUserMessage?: (content: string) => void;
  /** The page's resume view and handlers, passed through to the welcome card. */
  resume?: ResumeView | null;
  onUploadResume?: (file: File) => void;
  onRejectResume?: (message: string) => void;
  onRetryResumeStatus?: () => void;
}

export default function ChatArea({
  selectedAgent,
  userId,
  mode,
  threadId,
  loadedMessages,
  currentSection,
  onThreadIdChange,
  onSectionUpdate,
  onCompletionUpdate,
  onFirstUserMessage,
  resume,
  onUploadResume,
  onRejectResume,
  onRetryResumeStatus,
}: ChatAreaProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [copiedAll, setCopiedAll] = useState(false);

  const messagesPaneRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const isNearBottomRef = useRef(true);
  const hasUserMessageRef = useRef(false);
  const scrollFrameRef = useRef<number | null>(null);
  // Visual policy for this mounted thread, never another transcript/state cache.
  const revealMessageIdsRef = useRef(new Set<string>());

  const scheduleChatScroll = useCallback((trigger: ChatScrollTrigger) => {
    const behavior = chatScrollBehavior(trigger, isNearBottomRef.current);
    if (behavior === null) return;

    // Token events may arrive more than once per paint. Coalesce them into one
    // direct container scroll, and re-check the ref inside the frame so a manual
    // upward scroll that happened meanwhile wins.
    if (trigger === 'stream-token' && scrollFrameRef.current !== null) return;
    if (trigger !== 'stream-token' && scrollFrameRef.current !== null) {
      cancelAnimationFrame(scrollFrameRef.current);
      scrollFrameRef.current = null;
    }

    scrollFrameRef.current = requestAnimationFrame(() => {
      scrollFrameRef.current = null;
      const pane = messagesPaneRef.current;
      if (!pane) return;
      if (trigger === 'stream-token' && !isNearBottomRef.current) return;
      pane.scrollTo({ top: pane.scrollHeight, behavior });
    });
  }, []);

  const scrollRevealedText = useCallback(() => scheduleChatScroll('stream-token'), [scheduleChatScroll]);

  useEffect(() => () => {
    if (scrollFrameRef.current !== null) cancelAnimationFrame(scrollFrameRef.current);
  }, []);

  useEffect(() => {
    if (!isLoading) {
      inputRef.current?.focus();
    }
  }, [isLoading]);


  // Reset messages when threadId is null (reset button clicked)
  useEffect(() => {
    if (threadId === null) {
      revealMessageIdsRef.current.clear();
      setMessages([]);
      setInput('');
      setIsLoading(false);
      isNearBottomRef.current = true;
      hasUserMessageRef.current = false;

    }
  }, [threadId]);

  // Load messages when conversation is selected
  useEffect(() => {
    if (loadedMessages) {
      revealMessageIdsRef.current.clear();
      hasUserMessageRef.current = loadedMessages.some((message) => message.role === 'user');
      if (loadedMessages.length > 0) {
        setMessages(loadedMessages);
        isNearBottomRef.current = true;
        scheduleChatScroll('restored-history');
      }
    }
  }, [loadedMessages, threadId, scheduleChatScroll]);

  // No local transcript cache. The LangGraph checkpoint owns the conversation and
  // /api/history returns it; a copy here would be a cache nobody invalidates.



  const getAgentName = (_agentId: string) => 'JobBuddy';

  const getPlaceholderText = (_agentId: string) =>
    'Message JobBuddy about your career goals...';

  // Extract send message logic
  const handleSendMessage = useCallback(async (messageContent: string) => {
    if (!messageContent.trim() || isLoading || !selectedAgent || !userId) return;
    const requestStartedAt = performance.now();
    if (process.env.NODE_ENV === 'development') {
      console.debug('[JobBuddy timing] submit', { atMs: requestStartedAt });
    }

    const userMessage: Message = {
      id: Date.now().toString() + '-user',
      role: 'user',
      content: messageContent.trim()
    };

    if (!hasUserMessageRef.current) {
      hasUserMessageRef.current = true;
      onFirstUserMessage?.(userMessage.content);
    }

    // Sending is an explicit navigation action: reveal the new message once with
    // animation, then token growth below uses direct scrolls only.
    isNearBottomRef.current = true;
    // A follow-up finishes older visual reveals, without delaying the new turn.
    revealMessageIdsRef.current.clear();
    setMessages(prev => [...prev, userMessage]);
    scheduleChatScroll('new-user-message');
    setInput('');
    setIsLoading(true);

    try {
      const requestPayload = {
        messages: [{ role: 'user', content: userMessage.content }],
        userId: userId,
        threadId: threadId,
        mode: mode,
        agentId: selectedAgent
      };

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestPayload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle invoke mode
      if (mode === 'invoke') {
        const invokeData = await response.json();

        const assistantMessage: Message = {
          id: Date.now().toString() + '-assistant',
          role: 'assistant',
          content: invokeData.content
        };

        setMessages(prev => [...prev, assistantMessage]);

        if (invokeData.threadId && !threadId) {
          onThreadIdChange(invokeData.threadId);
        }
        if (invokeData.section) {
          onSectionUpdate(invokeData.section);
        }

        setIsLoading(false);
        return;
      }

      // Handle streaming mode
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      // A single SSE frame can straddle two network chunks, so the tail of a chunk is
      // carried forward rather than parsed as though it were a complete line. Without
      // this a token is dropped whenever a read happens to land mid-frame — rare
      // locally, much less rare over a real network.
      let carry = '';

      if (!reader) {
        throw new Error('No response body reader available');
      }

      const tempAssistantMessage: Message = {
        id: Date.now().toString() + '-assistant',
        role: 'assistant',
        content: ''
      };

      setMessages(prev => [...prev, tempAssistantMessage]);
      scheduleChatScroll('stream-token');
      // Synchronous running total. State is not readable mid-loop, and deriving the
      // text from a state updater is what broke streaming in the first place.
      let accumulated = '';
      // Every assistant line already on screen for this turn. A `message` event
      // repeating one of them is the backend echoing text that arrived as tokens;
      // a `message` event carrying anything else is a message the user would
      // otherwise only discover after a refresh.
      const shownThisTurn: string[] = [];
      // Extra bubbles appended after the streamed reply, so the token handler keeps
      // writing to the streamed bubble and never overwrites one of these.
      let extraBubbleCount = 0;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = carry + decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');
          carry = lines.pop() ?? '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') {
                // Stream is complete
                setIsLoading(false);
                return;
              }

              try {
                const parsed = JSON.parse(data);

                if (parsed.type === 'token') {
                  accumulated += parsed.content;
                  const nextContent = accumulated;
                  setMessages(prevMessages =>
                    prevMessages.map(msg =>
                      msg.id === tempAssistantMessage.id
                        ? { ...msg, content: nextContent }
                        : msg
                    )
                  );
                  scheduleChatScroll('stream-token');
                } else if (parsed.type === 'metadata') {
                  // Handle metadata event - extract thread_id
                  if (parsed.content && parsed.content.thread_id && !threadId) {
                    onThreadIdChange(parsed.content.thread_id);
                  }
                } else if (parsed.type === 'message') {
                  // A turn can persist more than one assistant message, and only
                  // the reply has tokens behind it. See planMessageEvent for why
                  // ignoring these outright made the readiness line appear only
                  // after a refresh.
                  const outcome = planMessageEvent(parsed.content, {
                    accumulated,
                    shown: shownThisTurn,
                    extraBubbles: extraBubbleCount,
                  });

                  if (outcome.kind === 'fill-placeholder') {
                    revealMessageIdsRef.current.add(tempAssistantMessage.id);
                    accumulated = outcome.text;
                    shownThisTurn.push(outcome.text.trim());
                    const nextContent = outcome.text;
                    setMessages(prevMessages =>
                      prevMessages.map(msg =>
                        msg.id === tempAssistantMessage.id
                          ? { ...msg, content: nextContent }
                          : msg
                      )
                    );
                    scheduleChatScroll('stream-token');
                  } else if (outcome.kind === 'append-bubble') {
                    extraBubbleCount += 1;
                    shownThisTurn.push(outcome.text.trim());
                    const extra: Message = {
                      id: `${tempAssistantMessage.id}-extra-${extraBubbleCount}`,
                      role: 'assistant',
                      content: outcome.text,
                    };
                    revealMessageIdsRef.current.add(extra.id);
                    setMessages(prevMessages => [...prevMessages, extra]);
                    scheduleChatScroll('stream-token');
                  }
                } else if (parsed.type === 'section') {
                  onSectionUpdate(parsed.content);
                } else if (parsed.type === 'completion') {
                  // PR6's public completion projection: the five canonical sections
                  // plus two booleans. Nothing internal crosses this boundary.
                  if (process.env.NODE_ENV === 'development') {
                    console.debug('[JobBuddy timing] completion-received', {
                      atMs: performance.now(), elapsedMs: performance.now() - requestStartedAt,
                    });
                  }
                  onCompletionUpdate?.(parsed.content as CompletionState);
                } else if (parsed.type === 'final_response') {
                  // Handle final response if needed
                  if (parsed.threadId && !threadId) {
                    onThreadIdChange(parsed.threadId);
                  }
                  if (parsed.section) {
                    onSectionUpdate(parsed.section);
                  }
                  setMessages(prevMessages =>
                    prevMessages.map(msg =>
                      msg.id === tempAssistantMessage.id
                        ? { ...msg, content: parsed.content }
                        : msg
                    )
                  );
                  scheduleChatScroll('stream-token');
                }
              } catch (e) {
                console.error('Parse error:', e);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
        // Ensure loading is set to false when stream ends
        setIsLoading(false);
      }

    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        id: Date.now().toString() + '-error',
        role: 'assistant',
        content: `Sorry, an error occurred while connecting to ${getAgentName(selectedAgent)}. Please try again.`
      };
      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
      // Stop auto mode on error
    }
  }, [
    isLoading,
    selectedAgent,
    userId,
    threadId,
    mode,
    onThreadIdChange,
    onSectionUpdate,
    onCompletionUpdate,
    onFirstUserMessage,
    scheduleChatScroll,
  ]);


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await handleSendMessage(input);
  };

  const canSendMessage = selectedAgent && userId && input.trim() && !isLoading;

  const copyToClipboard = async (text: string, messageId?: string) => {
    try {
      await navigator.clipboard.writeText(text);
      if (messageId) {
        setCopiedMessageId(messageId);
        setTimeout(() => setCopiedMessageId(null), 1500);
      }
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const copyAllConversation = async () => {
    const conversationText = messages.map(msg => {
      const role = msg.role === 'user' ? '👤 User' : `🤖 ${getAgentName(selectedAgent)}`;
      return `${role}:\n${msg.content}`;
    }).join('\n\n' + '='.repeat(50) + '\n\n');
    
    await copyToClipboard(conversationText);
    setCopiedAll(true);
    setTimeout(() => setCopiedAll(false), 1500);
  };

  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      minHeight: 0,
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 24px',
        borderBottom: '1px solid #e2e8f0',
        backgroundColor: 'white',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: '16px'
      }}>
         <div style={{ flex: 1 }}>
           <h1 style={{
             fontSize: '20px',
             fontWeight: 'bold',
             color: '#1e293b',
             margin: 0,
             display: 'flex',
             alignItems: 'center',
             gap: '8px'
           }}>
             {selectedAgent ? getAgentName(selectedAgent) : 'Select an Agent'}
           </h1>
           {selectedAgent && (
             <p style={{
               fontSize: '12px',
               color: '#64748b',
               margin: '4px 0 0 0'
             }}>
               AI Career Planning Agent
             </p>
           )}
         </div>

         {messages.length > 0 && (
           <button
             onClick={copyAllConversation}
             style={{
               padding: '8px 12px',
               backgroundColor: copiedAll ? '#059669' : '#10b981',
               color: 'white',
               border: 'none',
               borderRadius: '6px',
               fontSize: '12px',
               cursor: 'pointer',
               display: 'flex',
               alignItems: 'center',
               gap: '4px',
               transition: 'all 0.2s ease',
               transform: copiedAll ? 'scale(1.05)' : 'scale(1)',
               whiteSpace: 'nowrap'
             }}
             title={copiedAll ? "Copied all messages!" : "Copy all conversation"}
           >
             {copiedAll ? '✓ Copied!' : '📋 Copy All'}
           </button>
         )}
      </div>

      {/* Messages — the only scroll container in the app. */}
      <div
        ref={messagesPaneRef}
        onScroll={(event) => {
          const pane = event.currentTarget;
          isNearBottomRef.current = isNearChatBottom(pane);
        }}
        style={{
        flex: 1,
        minHeight: 0,
        overflowY: 'auto',
        padding: '24px',
        backgroundColor: '#f8fafc'
        }}
      >
        {!selectedAgent ? (
          <div style={{
            textAlign: 'center',
            paddingTop: '100px',
            color: '#64748b'
          }}>
            <div style={{ fontSize: '16px', marginBottom: '8px' }}>
              👈 Please select an agent from the left panel to start chatting
            </div>
          </div>
        ) : messages.length === 0 ? (
          <JobBuddyWelcome
            resume={resume}
            onUploadResume={onUploadResume}
            onRejectResume={onRejectResume}
            onRetryResumeStatus={onRetryResumeStatus}
          />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {messages.map((message) => (
              <div key={message.id} style={{
                display: 'flex',
                justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start'
              }}>
                <div style={{
                  maxWidth: '70%',
                  padding: '12px 16px',
                  borderRadius: '12px',
                  backgroundColor: message.role === 'user' ? '#3b82f6' : '#ffffff',
                  color: message.role === 'user' ? 'white' : '#1e293b',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '12px',
                    marginBottom: '6px',
                    minHeight: '20px'
                  }}>
                    <span style={{
                      fontSize: '10px',
                      fontWeight: '500',
                      opacity: 0.7,
                      minWidth: 0,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap'
                    }}>
                      {message.role === 'user' ? 'You' : getAgentName(selectedAgent)}
                    </span>
                    {message.role === 'assistant' && message.content && (
                      <button
                        onClick={() => copyToClipboard(message.content, message.id)}
                        style={{
                          flexShrink: 0,
                          width: '22px',
                          height: '22px',
                          border: 'none',
                          borderRadius: '4px',
                          backgroundColor: copiedMessageId === message.id ? '#10b981' : '#f1f5f9',
                          color: copiedMessageId === message.id ? 'white' : '#64748b',
                          cursor: 'pointer',
                          fontSize: '11px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          opacity: 0.7
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                        onMouseLeave={(e) => e.currentTarget.style.opacity = '0.7'}
                        title={copiedMessageId === message.id ? "Copied!" : "Copy message"}
                      >
                        {copiedMessageId === message.id ? '✓' : '📋'}
                      </button>
                    )}
                  </div>
                  <div style={{
                    fontSize: '14px',
                    lineHeight: '1.5',
                    whiteSpace: message.role === 'user' ? 'pre-wrap' : 'normal'
                  }}>
                    {message.role === 'assistant' && !message.content && isLoading ? (
                      // In-bubble, not a second card. The old standalone "Thinking..."
                      // block rendered alongside the streaming message, so one turn
                      // showed two assistant bubbles. This lives inside the one bubble
                      // and is replaced in place by the first token.
                      <span
                        style={{ display: 'inline-flex', alignItems: 'center', gap: 8, color: '#94a3b8' }}
                        aria-live="polite"
                      >
                        <span
                          style={{
                            width: 12,
                            height: 12,
                            border: '2px solid #e2e8f0',
                            borderTop: '2px solid #3b82f6',
                            borderRadius: '50%',
                            animation: 'spin 1s linear infinite'
                          }}
                        />
                      </span>
                    ) : message.role === 'assistant' ? (
                      <ProgressiveText
                        text={message.content}
                        animate={revealMessageIdsRef.current.has(message.id)}
                        onProgress={scrollRevealedText}
                      >
                        {(visibleContent) => (
                      <ReactMarkdown
                        components={{
                          p: ({ children }) => <p style={{ margin: '0 0 8px 0' }}>{children}</p>,
                          h1: ({ children }) => <h1 style={{ margin: '0 0 12px 0', fontSize: '18px', fontWeight: 'bold' }}>{children}</h1>,
                          h2: ({ children }) => <h2 style={{ margin: '0 0 10px 0', fontSize: '16px', fontWeight: 'bold' }}>{children}</h2>,
                          h3: ({ children }) => <h3 style={{ margin: '0 0 8px 0', fontSize: '15px', fontWeight: 'bold' }}>{children}</h3>,
                          ul: ({ children }) => <ul style={{ margin: '0 0 8px 0', paddingLeft: '16px' }}>{children}</ul>,
                          ol: ({ children }) => <ol style={{ margin: '0 0 8px 0', paddingLeft: '16px' }}>{children}</ol>,
                          li: ({ children }) => <li style={{ margin: '2px 0' }}>{children}</li>,
                          strong: ({ children }) => <strong style={{ fontWeight: 'bold' }}>{children}</strong>,
                          em: ({ children }) => <em style={{ fontStyle: 'italic' }}>{children}</em>,
                          code: ({ children }) => <code style={{ 
                            backgroundColor: '#f1f5f9', 
                            padding: '2px 4px', 
                            borderRadius: '3px', 
                            fontSize: '13px',
                            fontFamily: 'monospace'
                          }}>{children}</code>,
                          pre: ({ children }) => <pre style={{ 
                            backgroundColor: '#f1f5f9', 
                            padding: '8px', 
                            borderRadius: '6px', 
                            overflow: 'auto',
                            margin: '8px 0'
                          }}>{children}</pre>,
                          blockquote: ({ children }) => <blockquote style={{ 
                            borderLeft: '3px solid #cbd5e1', 
                            paddingLeft: '12px', 
                            margin: '8px 0',
                            fontStyle: 'italic'
                          }}>{children}</blockquote>
                        }}
                      >
                        {visibleContent}
                      </ReactMarkdown>
                        )}
                      </ProgressiveText>
                    ) : (
                      message.content
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Composer — a sibling of the scroll container, never inside it. */}
      <div style={{
        flexShrink: 0,
        padding: '16px 24px',
        backgroundColor: 'white',
        borderTop: '1px solid #e2e8f0'
      }}>
        <form
          onSubmit={handleSubmit}
          style={{ display: 'flex', gap: '12px', alignItems: 'stretch', minHeight: '46px' }}
        >
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={getPlaceholderText(selectedAgent)}
            disabled={!selectedAgent || !userId}
            style={{
              flex: 1,
              padding: '12px 16px',
              border: '1px solid #d1d5db',
              borderRadius: '8px',
              fontSize: '14px',
              outline: 'none',
              backgroundColor: 'white',
              cursor: 'text'
            }}
          />
          <button
            type="submit"
            disabled={!canSendMessage}
            style={{
              padding: '12px 24px',
              backgroundColor: canSendMessage ? '#3b82f6' : '#9ca3af',
              color: 'white',
              borderRadius: '8px',
              border: 'none',
              fontSize: '14px',
              cursor: canSendMessage ? 'pointer' : 'not-allowed',
              fontWeight: '500'
            }}
          >
            Send
          </button>
        </form>
      </div>

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}
