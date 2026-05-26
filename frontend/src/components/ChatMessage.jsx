import { useState } from "react"

function ChatMessage({ message }) {
  const [showSources, setShowSources] = useState(false)

  return (
    <div className={`message ${message.role}`}>
      <span className="message-label">
        {message.role === "user" ? "You" : "Cricket Agent"}
      </span>

      {message.isError ? (
        <div className="error-bubble">{message.content}</div>
      ) : (
        <div className="message-bubble">{message.content}</div>
      )}

      {message.sources && message.sources.length > 0 && (
        <>
          <button
            className="sources-toggle"
            onClick={() => setShowSources(!showSources)}
          >
            {showSources
              ? "Hide sources"
              : `${message.sources.length} sources used`}
          </button>

          {showSources && (
            <div className="sources-list">
              {message.sources.map((source, index) => (
                <div key={index} className="source-item">
                  <strong>#{index + 1}</strong> {source}
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