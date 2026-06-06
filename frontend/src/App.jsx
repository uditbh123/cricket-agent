const sendMessage = async (question) => {
    const userMessage = { role: "user", content: question }

    // Capture current messages before state update
    const currentMessages = messages

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    const assistantIndex = currentMessages.length + 1

    setMessages(prev => [...prev, {
        role: "assistant",
        content: "",
        sources: [],
        isStreaming: true,
    }])

    try {
        const response = await fetch(`${API_URL}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                // Send last 6 messages so backend has conversation context
                // Filter out isStreaming/isError flags — backend only needs role + content
                history: currentMessages.slice(-6).map(m => ({
                    role: m.role,
                    content: m.content
                }))
            }),
        })

        if (!response.ok) throw new Error("Server error")

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ""

        while (true) {
            const { done, value } = await reader.read()
            if (done) break

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split("\n")
            buffer = lines.pop()

            for (const line of lines) {
                if (!line.trim()) continue
                try {
                    const parsed = JSON.parse(line)

                    if (parsed.type === "sources") {
                        setMessages(prev => prev.map((msg, i) =>
                            i === assistantIndex
                                ? { ...msg, sources: parsed.data }
                                : msg
                        ))
                    } else if (parsed.type === "token") {
                        setMessages(prev => prev.map((msg, i) =>
                            i === assistantIndex
                                ? { ...msg, content: msg.content + parsed.data }
                                : msg
                        ))
                    } else if (parsed.type === "done") {
                        setMessages(prev => prev.map((msg, i) =>
                            i === assistantIndex
                                ? { ...msg, isStreaming: false }
                                : msg
                        ))
                    }
                } catch { }
            }
        }

    } catch (error) {
        setMessages(prev => prev.map((msg, i) =>
            i === assistantIndex
                ? {
                    ...msg,
                    content: "Could not reach the backend. Make sure FastAPI is running on port 8000.",
                    isError: true,
                    isStreaming: false,
                }
                : msg
        ))
    } finally {
        setIsLoading(false)
    }
}