import { useEffect, useRef } from "react"
import ChatMessage from "./ChatMessage"

function ChatWindow({ messages, isLoading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

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

  // Show typing indicator only when loading AND the last message is
  // from the user (i.e. assistant reply hasn't started yet).
  // Once the assistant message exists (even empty/streaming), we stop
  // showing the dots — the streaming cursor takes over.
  const lastMessage = messages[messages.length - 1]
  const showTyping = isLoading && lastMessage?.role === "user"

  return (
    <div className="chat-window">
      {messages.map(message => (
        <ChatMessage key={message.id} message={message} />
      ))}

      {showTyping && (
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