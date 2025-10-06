import React, { useState, useEffect, useRef } from "react";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { SendHorizontal } from "lucide-react";

const Chat = () => {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const messagesEndRef = useRef(null);

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    const newMessage = { question, answer: "Thinking..." };
    setMessages((prev) => [...prev, newMessage]);
    const currentIndex = messages.length;
    setQuestion("");

    try {
      const response = await fetch("http://192.168.11.111:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question }),
      });

      const data = await response.json();

      const formattedAnswer = data.answer
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => (line.startsWith("*") ? line : `* ${line}`))
        .join("\n");

      setMessages((prev) => {
        const updated = [...prev];
        updated[currentIndex] = { question: newMessage.question, answer: formattedAnswer };
        return updated;
      });
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[currentIndex] = { question: newMessage.question, answer: "Error fetching answer" };
        return updated;
      });
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex flex-col h-screen bg-gray-50">
        <div className="flex items-center gap-4 px-10 py-4">
            <img
                src="/logo1.png"
                alt="logo"
                className="w-[100px]"
              />
            <h2 className="text-4xl text-[#1a83bc] font-bold">Chatbot</h2>
        </div>

      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto rounded-lg px-10 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className="space-y-2">
            {/* User bubble */}
            <div className="flex justify-end items-start space-x-2">
              <div className="bg-[#1a83bccc] text-white px-4 py-2 rounded-2xl max-w-lg">
                {msg.question}
              </div>
              <img
                src="/user.png"
                alt="User"
                className="w-10 h-10 rounded-full"
              />
            </div>

            {/* Bot bubble */}
            <div className="flex justify-start items-start space-x-2">
              <img
                src="/robot.png"
                alt="Bot"
                className="w-10 h-10 rounded-full"
              />
              <div className="bg-[#1a83bc1a] text-gray-900 px-4 py-2 rounded-2xl max-w-5xl">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {msg.answer}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input box */}
      <form
        onSubmit={handleAsk}
        className="flex items-center gap-2 mt-4 border-t border-gray-300 px-10 py-4"
      >
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question..."
          className="flex-1 bg-white text-black border border-gray-300 rounded-md px-3 py-2 outline-none"
        />
        <button
          type="submit"
          className="bg-[#1a83bccc] text-white px-4 py-2 rounded-md"
        >
          <SendHorizontal />
        </button>
      </form>
    </div>
  );
};

export default Chat;
 