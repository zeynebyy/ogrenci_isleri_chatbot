"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bot,
  BookOpenCheck,
  CheckCircle2,
  ChevronRight,
  GraduationCap,
  Info,
  MessageSquareText,
  Moon,
  PanelRightOpen,
  Send,
  ShieldCheck,
  Sparkles,
  Sun,
  Trash2,
  User,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

interface Source {
  source: string;
  madde: string;
  title?: string;
}

interface ChatResponse {
  answer?: string;
  sources?: Source[];
  error?: string | null;
}

interface Message {
  role: "user" | "bot";
  content: string;
  id: string;
}

type Theme = "light" | "dark";

const API_URL = "http://127.0.0.1:8002/api/chat";
const toApiRole = (role: Message["role"]) => (role === "user" ? "user" : "assistant");

const quickQuestions = [
  "Staj kaç gün yapılır?",
  "Muafiyet için hangi belgeler gerekir?",
  "Erasmus için dil sınavı şartı var mı?",
  "Çift anadal başvuru şartları nelerdir?",
  "Yaz okulundan ders alabilir miyim?",
  "Özel öğrenci kimdir?",
];

const capabilities = [
  { label: "19 resmi yönerge", icon: BookOpenCheck },
  { label: "Kaynak ve madde gösterimi", icon: ShieldCheck },
  { label: "Takip sorusu bağlamı", icon: MessageSquareText },
];

function formatResponse(data: ChatResponse) {
  const answer = data.answer?.trim() || "Bu soru için cevap üretilemedi.";
  const hasSourceInAnswer = /kaynak\s*:/i.test(answer);
  const sources = data.sources ?? [];

  if (hasSourceInAnswer || sources.length === 0) {
    return answer;
  }

  const sourceText = sources
    .map((source) => `${source.source}, Madde ${source.madde}`)
    .join("; ");
  return `${answer}\n\nKaynak: ${sourceText}.`;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "bot",
      content:
        "Merhaba. Ben BEÜ Öğrenci İşleri Asistanı. Resmi yönergelerdeki bilgilere göre sorularını kaynak göstererek yanıtlayabilirim.",
      id: "initial",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [theme, setTheme] = useState<Theme>("light");
  const messagesAreaRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const savedTheme = localStorage.getItem("beu-chat-theme") as Theme | null;
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(savedTheme ?? (prefersDark ? "dark" : "light"));
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("beu-chat-theme", theme);
  }, [theme]);

  useEffect(() => {
    const messagesArea = messagesAreaRef.current;
    if (messagesArea) {
      messagesArea.scrollTop = messagesArea.scrollHeight;
    }
  }, [messages]);

  const updateBotMessage = (id: string, content: string) => {
    setMessages((prev) =>
      prev.map((message) => (message.id === id ? { ...message, content } : message)),
    );
  };

  const sendQuestion = async (question: string) => {
    const cleanQuestion = question.trim();
    if (!cleanQuestion || isLoading) return;

    const userMessage: Message = {
      role: "user",
      content: cleanQuestion,
      id: crypto.randomUUID(),
    };
    const botMessageId = crypto.randomUUID();

    setMessages((prev) => [
      ...prev,
      userMessage,
      { role: "bot", content: "", id: botMessageId },
    ]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json; charset=utf-8" },
        body: JSON.stringify({
          message: cleanQuestion,
          history: messages
            .filter((message) => message.id !== "initial" && message.content.trim())
            .slice(-8)
            .map((message) => ({
              role: toApiRole(message.role),
              content: message.content,
            })),
        }),
      });

      if (!response.ok) {
        throw new Error(`API ${response.status}`);
      }

      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        const data = (await response.json()) as ChatResponse;
        updateBotMessage(botMessageId, formatResponse(data));
      } else {
        const text = await response.text();
        updateBotMessage(botMessageId, text.trim() || "Bu soru için cevap üretilemedi.");
      }
    } catch (error) {
      console.error("Chat error:", error);
      updateBotMessage(
        botMessageId,
        "Backend sunucusuna ulaşılamadı. Lütfen API'nin 8002 portunda çalıştığını kontrol edin.",
      );
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        role: "bot",
        content: "Sohbet sıfırlandı. Yeni sorunuzu yazabilirsiniz.",
        id: crypto.randomUUID(),
      },
    ]);
  };

  const toggleTheme = () => {
    setTheme((currentTheme) => (currentTheme === "light" ? "dark" : "light"));
  };

  return (
    <main className="app-shell">
      <section className="workspace">
        <aside className="insight-panel">
          <div className="brand-card">
            <div className="brand-mark">
              <GraduationCap size={30} />
            </div>
            <div>
              <div className="eyebrow">
                <Sparkles size={14} />
                Bitlis Eren Üniversitesi
              </div>
              <h1>BEÜ Asistan</h1>
              <p>Öğrenci işleri yönergeleri için belge tabanlı danışman.</p>
            </div>
          </div>

          <div className="status-card">
            <div className="status-line">
              <span className="status-dot" />
              Resmi belge modunda
            </div>
            <p>Yanıtlar yönerge parçaları ve madde bilgileriyle desteklenir.</p>
          </div>

          <div className="capability-list">
            {capabilities.map(({ label, icon: Icon }) => (
              <div className="capability" key={label}>
                <Icon size={18} />
                <span>{label}</span>
              </div>
            ))}
          </div>

          <div className="quick-panel">
            <div className="panel-title">
              <PanelRightOpen size={17} />
              Hızlı sorular
            </div>
            <div className="quick-list">
              {quickQuestions.map((question) => (
                <button
                  className="quick-question"
                  disabled={isLoading}
                  key={question}
                  onClick={() => sendQuestion(question)}
                >
                  <span>{question}</span>
                  <ChevronRight size={16} />
                </button>
              ))}
            </div>
          </div>
        </aside>

        <section className="chat-panel">
          <header className="chat-header">
            <div>
              <div className="header-kicker">
                <CheckCircle2 size={16} />
                RAG destekli cevap üretimi
              </div>
              <h2>Öğrenci işleri sohbet ekranı</h2>
            </div>

            <div className="header-actions">
              <button
                onClick={toggleTheme}
                className="icon-button"
                title={theme === "light" ? "Koyu moda geç" : "Açık moda geç"}
                aria-label={theme === "light" ? "Koyu moda geç" : "Açık moda geç"}
              >
                {theme === "light" ? <Moon size={19} /> : <Sun size={19} />}
              </button>
              <button onClick={clearChat} className="icon-button danger" title="Sohbeti temizle">
                <Trash2 size={19} />
              </button>
            </div>
          </header>

          <div ref={messagesAreaRef} className="messages-area">
            <AnimatePresence initial={false}>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 12, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 8, scale: 0.98 }}
                  className={`message-row ${message.role}`}
                >
                  <div className="avatar">
                    {message.role === "user" ? <User size={16} /> : <Bot size={16} />}
                  </div>
                  <div className="message-stack">
                    <span className="message-author">
                      {message.role === "user" ? "Siz" : "BEÜ Asistan"}
                    </span>
                    <div className="message-bubble">
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                      {message.role === "bot" && !message.content && isLoading && (
                        <div className="typing-indicator" aria-label="Cevap yazılıyor">
                          <span />
                          <span />
                          <span />
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>

          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault();
              sendQuestion(input);
            }}
          >
            <input
              type="text"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Staj, muafiyet, Erasmus, yaz okulu veya diploma işlemleri hakkında sorun..."
              disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !input.trim()} title="Gönder">
              <Send size={18} />
            </button>
          </form>

          <footer className="chat-footer">
            <Info size={13} />
            Kişiye özel OBS, transkript veya dilekçe sonucu bilgileri için ilgili birime başvurunuz.
          </footer>
        </section>
      </section>
    </main>
  );
}
