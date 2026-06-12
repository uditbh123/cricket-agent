import { useEffect, useRef } from "react"
import ChatMessage from "./ChatMessage"

const SUGGESTIONS = [
  "Who has scored the most Test centuries?",
  "How does the Duckworth-Lewis method work?",
  "Tell me about the Nepal Premier League",
  "Who is the fastest bowler ever?",
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
          <div className="empty-logo" />
          <h2>Cricket Agent</h2>
          <p>
            Ask me anything about cricket — players, rules,
            records, tournaments, history.
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
        <div className="typing-wrap">
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