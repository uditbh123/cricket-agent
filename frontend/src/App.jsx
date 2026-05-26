import { useState } from "react"
import axios from "axios"
import ChatWindow from "./components/ChatWindow"
import ChatInput from "./components/ChatInput"

const API_URL = "http://localhost:8000"

function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const sendMessage = async (question) => {
    setMessages(prev => [...prev, {
      role: "user",
      content: question
    }])
    setIsLoading(true)

    try {
      const response = await axios.post(`${API_URL}/ask`, { question })

      setMessages(prev => [...prev, {
        role: "assistant",
        content: response.data.answer,
        sources: response.data.sources,
      }])
    } catch (error) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "Could not reach the backend. Make sure the FastAPI server is running on port 8000.",
        isError: true,
        sources: [],
      }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="header">
        <div className="header-icon">🏏</div>
        <div className="header-text">
          <h1>Cricket Agent</h1>
          <p>Ask me anything about cricket. I answer from Wikipedia.</p>        
        </div>
        <div className="status-dot" title="Backend connected" />
      </div>

      <ChatWindow
        messages={messages}
        isLoading={isLoading}
        onSuggestion={sendMessage}
      />

      <ChatInput
        onSend={sendMessage}
        isLoading={isLoading}
      />
    </div>
  )
}

export default App