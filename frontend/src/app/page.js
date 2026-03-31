"use client";
import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { motion, useScroll, useSpring } from "framer-motion";
import { TrendingUp, Activity, BarChart3, Globe, MessageSquare } from "lucide-react";

// Dynamically import Scene3D to avoid SSR issues with Canvas/Three.js
const Scene3D = dynamic(() => import("@/components/Scene3D"), { ssr: false });

export default function Home() {
  const [scrollY, setScrollY] = useState(0);
  const [marketData, setMarketData] = useState(null);
  const [prediction, setPrediction] = useState("");
  
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001
  });

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", handleScroll);
    
    // Fetch initial market data
    fetch("http://localhost:8000/api/market-status")
      .then(res => res.json())
      .then(data => setMarketData(data))
      .catch(err => console.error("Backend not running yet:", err));
      
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    const text = e.target.news.value;
    const res = await fetch(`http://localhost:8000/api/predict?text=${encodeURIComponent(text)}`, {
      method: 'POST'
    });
    const data = await res.json();
    setPrediction(data);
  };

  return (
    <main className="relative min-h-[500vh]">
      {/* 3D Background */}
      <Scene3D scrollY={scrollY} />
      
      {/* Progress Bar */}
      <motion.div className="fixed top-0 left-0 right-0 h-1 bg-[#00d4ff] z-50 origin-left" style={{ scaleX }} />

      {/* Hero Section */}
      <section className="h-screen flex flex-col items-center justify-center text-center px-4">
        <motion.h1 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-6xl md:text-8xl font-bold tracking-tighter text-glow mb-4"
        >
          MARKET <span className="text-white opacity-50">REGIME</span> AI
        </motion.h1>
        <motion.p 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.8 }}
          className="text-xl text-gray-400 max-w-2xl"
        >
          An immersive multi-modal intelligence dashboard. Scroll to travel through the data void.
        </motion.p>
        <motion.div 
          animate={{ y: [0, 10, 0] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="mt-20 text-gray-500 uppercase text-xs tracking-widest"
        >
          Scroll to Explore
        </motion.div>
      </section>

      {/* Data Section 1: Real-time Stats */}
      <section className="h-screen flex items-center justify-center px-4">
        <motion.div 
          initial={{ opacity: 0, x: -50 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ margin: "-100px" }}
          className="glass-card p-8 rounded-3xl max-w-4xl w-full grid grid-cols-1 md:grid-cols-3 gap-8"
        >
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-[#00d4ff]">
              <TrendingUp size={20} />
              <span className="text-xs font-bold uppercase tracking-widest">Market Price</span>
            </div>
            <h3 className="text-4xl font-bold">₹{marketData?.price?.toLocaleString() || "73,450"}</h3>
            <p className="text-sm text-gray-500">BSE SENSEX Index</p>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-purple-400">
              <Activity size={20} />
              <span className="text-xs font-bold uppercase tracking-widest">AI Regime</span>
            </div>
            <h3 className="text-4xl font-bold">{marketData?.regime || "Bull"}</h3>
            <p className="text-sm text-gray-500">Fusion Model Analysis</p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2 text-green-400">
              <BarChart3 size={20} />
              <span className="text-xs font-bold uppercase tracking-widest">Volatility</span>
            </div>
            <h3 className="text-4xl font-bold">{(marketData?.volatility * 100).toFixed(1) || "12.4"}%</h3>
            <p className="text-sm text-gray-500">Annualized Risk</p>
          </div>
        </motion.div>
      </section>

      {/* Data Section 2: Macro intelligence */}
      <section className="h-screen flex items-center justify-center px-4">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ margin: "-100px" }}
          className="glass-card p-8 rounded-3xl max-w-lg w-full text-center space-y-6"
        >
          <Globe className="mx-auto text-[#00d4ff]" size={48} />
          <h2 className="text-3xl font-bold">Macro Sensitivity</h2>
          <p className="text-gray-400 italic">
            "The system is currently sensitive to Crude Oil and VIX fluctuations, driving the final fusion prediction."
          </p>
          <div className="flex justify-between items-center px-4 py-3 bg-white/5 rounded-xl text-sm">
            <span>RSI Momentum</span>
            <span className="text-[#00d4ff] font-mono">{marketData?.rsi?.toFixed(1) || "54.2"}</span>
          </div>
        </motion.div>
      </section>

      {/* Data Section 3: NLP Prediction */}
      <section className="h-screen flex items-center justify-center px-4">
        <motion.div 
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ margin: "-100px" }}
          className="glass-card p-10 rounded-3xl max-w-2xl w-full space-y-8"
        >
          <div className="flex items-center gap-3">
            <MessageSquare className="text-purple-400" />
            <h2 className="text-2xl font-bold tracking-tight">AI News Predictor</h2>
          </div>
          
          <form onSubmit={handlePredict} className="space-y-4">
            <textarea 
              name="news"
              placeholder="Paste latest market headlines here..."
              className="w-full bg-black/50 border border-white/10 rounded-2xl p-4 text-sm focus:outline-none focus:border-[#00d4ff] transition-colors"
              rows={4}
            />
            <button 
              type="submit"
              className="w-full py-4 bg-[#00d4ff] text-black font-bold rounded-2xl hover:bg-white hover:scale-[1.02] active:scale-[0.98] transition-all"
            >
              Analyze with Deep Fusion
            </button>
          </form>

          {prediction && (
             <motion.div 
               initial={{ opacity: 0, height: 0 }}
               animate={{ opacity: 1, height: 'auto' }}
               className="p-6 bg-white/5 rounded-2xl border border-[#00d4ff]/20"
             >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs uppercase tracking-widest text-[#00d4ff]">Result</span>
                  <span className="font-mono text-xl">{prediction.prediction}</span>
                </div>
                <div className="text-xs text-gray-500">
                  Confidence: {(prediction.confidence * 100).toFixed(1)}% | Sentiment: {prediction.sentiment_score.toFixed(3)}
                </div>
             </motion.div>
          )}
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="h-[20vh] flex items-center justify-center text-gray-600 text-xs uppercase tracking-[0.5em]">
        End of Void / Next-Gen AI
      </footer>
    </main>
  );
}
