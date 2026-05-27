import { useEffect, useRef } from "react"
import ChatMessage from "./ChatMessage"

const SUGGESTIONS = [
  "What are the Laws of Cricket?",
  "How does Duckworth-Lewis work?",
  "Who has the most Test centuries?",
  "What is the difference between Test and T20?",
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
            Ask me anything about cricket — rules, players,
            tournaments, records, or history.
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

      {isLoading && messages[messages.length - 1]?.role !== "assistant" && (
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