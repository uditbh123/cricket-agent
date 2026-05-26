import { useEffect, useRef } from "react"
import ChatMessage from "./ChatMessage"

const SUGGESTIONS = [
  "What are the Laws of Cricket?",
  "How does the Duckworth-Lewis method work?",
  "Who has scored the most Test centuries?",
  "What is the difference between Test and ODI cricket?",
]

function ChatWindow({ messages, isLoading, onSuggestion }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isLoading])

  if (messages.length === 0) {
    return (
      <div className="chat-window">
        <div className="empty-state">
          <div className="empty-icon">🏏</div>
          <h2>Cricket Knowledge Agent</h2>
          <p>
            Ask me anything about cricket — rules, players, tournaments,
            records, or history. I answer from Wikipedia.
          </p>
          <div className="suggestions">
            {SUGGESTIONS.map((s, i) => (
              <button
                key={i}
                className="suggestion-btn"
                onClick={() => onSuggestion(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-window">
      {messages.map((message, index) => (
        <ChatMessage key={index} message={message} />
      ))}

      {isLoading && (
        <div className="message assistant">
          <span className="message-label">Cricket Agent</span>
          <div className="typing-indicator">
            <span /><span /><span />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}

export default ChatWindow