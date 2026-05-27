import { useState, useRef } from "react"

const MAX_CHARS = 300

function ChatInput({ onSend, isLoading }) {
  const [input, setInput] = useState("")
  const textareaRef = useRef(null)

  const handleSend = () => {
    if (!input.trim() || isLoading) return
    onSend(input.trim())
    setInput("")
    if (textareaRef.current) {
      textareaRef.current.style.height = "46px"
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleChange = (e) => {
    if (e.target.value.length > MAX_CHARS) return
    setInput(e.target.value)
    const ta = textareaRef.current
    if (ta) {
      ta.style.height = "46px"
      ta.style.height = Math.min(ta.scrollHeight, 130) + "px"
    }
  }

  const remaining = MAX_CHARS - input.length
  const charClass =
    remaining < 20 ? "danger" : remaining < 60 ? "warning" : ""

  return (
    <div className="chat-input-area">
      <div className="input-row">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask a cricket question... (Enter to send, Shift+Enter for new line)"
          disabled={isLoading}
          rows={1}
        />
        <button
          className="send-button"
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
          title="Send"
        >
          ↑
        </button>
      </div>
      <div className="input-meta">
        <span className={`char-count ${charClass}`}>
          {remaining < 100 ? `${remaining} left` : ""}
        </span>
      </div>
    </div>
  )
}

export default ChatInput