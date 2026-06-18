import { useState, useCallback } from "react"
import ChatWindow from "./components/ChatWindow"
import ChatInput from "./components/ChatInput"

const API_URL = "http://localhost:8000"

let _nextId = 0
const newId = () => String(++_nextId)

function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const sendMessage = useCallback(async (question) => {
    if (isLoading) return

    // Snapshot history BEFORE adding new messages
    const historySnapshot = messages

    const userMsg = { id: newId(), role: "user", content: question }
    const assistantId = newId()
    const assistantMsg = {
      id: assistantId,
      role: "assistant",
      content: "",
      sources: [],
      isStreaming: true,
      isError: false,
    }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setIsLoading(true)

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          history: historySnapshot.slice(-12).map(m => ({
            role: m.role,
            content: m.content,
          })),
        }),
      })

      if (!response.ok) throw new Error(`Server error ${response.status}`)

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      // Helper: update the assistant message by its stable ID
      const updateAssistant = (updater) =>
        setMessages(prev =>
          prev.map(m => m.id === assistantId ? { ...m, ...updater(m) } : m)
        )

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() // keep incomplete last line in buffer

        for (const line of lines) {
          if (!line.trim()) continue
          try {
            const event = JSON.parse(line)
            if (event.type === "sources") {
              updateAssistant(() => ({ sources: event.data }))
            } else if (event.type === "token") {
              updateAssistant(m => ({ content: m.content + event.data }))
            } else if (event.type === "done") {
              updateAssistant(() => ({ isStreaming: false }))
            }
          } catch {
            // malformed line — skip silently
          }
        }
      }

      // Ensure streaming is marked done even if "done" event was missing
      updateAssistant(() => ({ isStreaming: false }))

    } catch (err) {
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantId
            ? {
                ...m,
                content: "Could not reach the backend. Make sure FastAPI is running on port 8000.",
                isError: true,
                isStreaming: false,
              }
            : m
        )
      )
    } finally {
      setIsLoading(false)
    }
  }, [messages, isLoading])

  const clearChat = useCallback(() => {
    if (messages.length === 0) return
    setMessages([])
  }, [messages.length])

  return (
    <>
      <div className="bg-animated">
        <div className="bg-grid" />
        <div className="bg-orb bg-orb-1" />
        <div className="bg-orb bg-orb-2" />
        <div className="bg-orb bg-orb-3" />
      </div>

      <div className="app">
        <div className="header">
          <div className="header-brand">
            <div className="logo" />
            <h1>Cricket Agent</h1>
          </div>
          <div className="header-actions">
            {messages.length > 0 && (
              <button className="clear-btn" onClick={clearChat}>
                New chat
              </button>
            )}
          </div>
        </div>

        <ChatWindow messages={messages} isLoading={isLoading} />

        <ChatInput onSend={sendMessage} isLoading={isLoading} />
      </div>
    </>
  )
}

export default App