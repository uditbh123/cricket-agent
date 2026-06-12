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

  return (
    <div className={`message-group ${message.role}`}>
      <span className="message-label">
        {message.role === "user" ? "You" : "Cricket Agent"}
      </span>

      {message.role === "user" ? (
        <div className="user-bubble">{message.content}</div>
      ) : message.isError ? (
        <div className="error-bubble">{message.content}</div>
      ) : (
        <div className="assistant-bubble">
          <div className="md-content">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
          {message.isStreaming && (
            <span className="streaming-cursor" />
          )}
        </div>
      )}

      {message.role === "assistant" && !message.isError && !message.isStreaming && (
        <div className="message-actions">
          <button
            className={`action-btn ${copied ? "copied" : ""}`}
            onClick={handleCopy}
          >
            {copied ? "✓ Copied" : "Copy"}
          </button>

          {message.sources && message.sources.length > 0 && (
            <button
              className="sources-toggle"
              onClick={() => setShowSources(!showSources)}
            >
              {showSources ? "Hide sources" : `${message.sources.length} sources`}
            </button>
          )}
        </div>
      )}

      {showSources && message.sources && (
        <div className="sources-panel">
          <div className="sources-panel-header">
            Sources used
          </div>
          {message.sources.map((source, i) => (
            <div key={i} className="source-item">
              {source.slice(0, 200)}{source.length > 200 ? "..." : ""}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ChatMessage