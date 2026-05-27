import { useState } from "react"

function ChatMessage({ message }) {
  const [showSources, setShowSources] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={`message ${message.role}`}>
      <span className="message-label">
        {message.role === "user" ? "You" : "Cricket Agent"}
      </span>

      {message.isError ? (
        <div className="error-bubble">{message.content}</div>
      ) : (
        <div className="message-bubble">
          {message.content}
          {message.isStreaming && (
            <span className="streaming-cursor" />
          )}
        </div>
      )}

      {message.role === "assistant" && !message.isError && !message.isStreaming && (
        <div className="message-actions">
          <button
            className={`copy-btn ${copied ? "copied" : ""}`}
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
        <div className="sources-list">
          {message.sources.map((source, index) => (
            <div key={index} className="source-item">
              <strong style={{ color: "#444" }}>#{index + 1}</strong>{" "}
              {source}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ChatMessage