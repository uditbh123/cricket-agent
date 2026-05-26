import { useState, useRef } from "react"

function ChatInput({ onSend, isLoading }) {
  const [input, setInput] = useState("")
  const textareaRef = useRef(null)

  const handleSend = () => {
    if (!input.trim() || isLoading) return
    onSend(input.trim())
    setInput("")
    textareaRef.current.style.height = "46px"
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInput = (e) => {
    setInput(e.target.value)
    const ta = textareaRef.current
    ta.style.height = "46px"
    ta.style.height = Math.min(ta.scrollHeight, 130) + "px"
  }

  return (
    <div className="chat-input-area">
      <textarea
        ref={textareaRef}
        value={input}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        placeholder="Ask a cricket question... (Enter to send)"
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
  )
}

export default ChatInput