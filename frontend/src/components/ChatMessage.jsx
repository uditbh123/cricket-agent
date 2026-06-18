import { useState } from "react"
import ReactMarkdown from "react-markdown"

function ChatMessage({ message }) {
  const [showSources, setShowSources] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (message.role === "user") {
    return (
      <div className="message-group user">
        <span className="message-label">You</span>
        <div className="user-bubble">{message.content}</div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className="message-group assistant">
      <span className="message-label">Cricket Agent</span>

      {message.isError ? (
        <div className="error-bubble">{message.content}</div>
      ) : (
        <div className="assistant-bubble">
          <div className="md-content">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
          {message.isStreaming && <span className="streaming-cursor" />}
        </div>
      )}

      {/* Actions only shown when streaming is done and no error */}
      {!message.isStreaming && !message.isError && (
        <>
          <div className="message-actions">
            <button
              className={`action-btn ${copied ? "copied" : ""}`}
              onClick={handleCopy}
            >
              {copied ? "✓ Copied" : "Copy"}
            </button>

            {message.sources?.length > 0 && (
              <button
                className="sources-toggle"
                onClick={() => setShowSources(v => !v)}
              >
                {showSources ? "Hide sources" : `${message.sources.length} sources`}
              </button>
            )}
          </div>

          {showSources && (
            <div className="sources-panel">
              <div className="sources-panel-header">Sources used</div>
              {message.sources.map((source, i) => (
                <div key={i} className="source-item">
                  {source.length > 200 ? source.slice(0, 200) + "…" : source}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default ChatMessage