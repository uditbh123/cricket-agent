import { useState, useRef } from "react"

const MAX_CHARS = 500

function ChatInput({ onSend, isLoading }) {
  const [input, setInput] = useState("")
  const textareaRef = useRef(null)

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed || isLoading) return
    onSend(trimmed)
    setInput("")
    if (textareaRef.current) {
      textareaRef.current.style.height = "24px"
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
      ta.style.height = "24px"
      ta.style.height = Math.min(ta.scrollHeight, 160) + "px"
    }
  }

  const remaining = MAX_CHARS - input.length
  const charClass = remaining < 30 ? "danger" : remaining < 80 ? "warning" : ""

  return (
    <div className="input-area">
      <div className="input-container">
        <div className="input-row">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything about cricket..."
            disabled={isLoading}
            rows={1}
          />
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            title="Send"
          >
            ↑
          </button>
        </div>
        <div className="input-footer">
          <span className="input-hint">Enter to send · Shift+Enter for new line</span>
          <span className={`char-count ${charClass}`}>
            {remaining < 150 ? remaining : ""}
          </span>
        </div>
      </div>
    </div>
  )
}

export default ChatInput