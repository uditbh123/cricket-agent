import { useEffect, useRef } from "react"
import ChatMessage from "./ChatMessage"

function ChatWindow({ messages, isLoading, onSuggestion }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isLoading])

  if (messages.length === 0) {
    return (
      <div className="chat-window" style={{ overflow: "hidden" }}>
        <div className="empty-state">
          <div className="empty-logo" />
          <h2>What do you want to know?</h2>
          <p>Ask me anything about cricket</p>
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