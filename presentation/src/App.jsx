import { useEffect } from 'react'
import Reveal from 'reveal.js'
import 'reveal.js/dist/reveal.css'
import 'reveal.js/dist/theme/black.css'
import 'reveal.js/plugin/highlight/monokai.css'
import Highlight from 'reveal.js/plugin/highlight/highlight.esm.js'

// 슬라이드 컴포넌트들
import TitleSlide from './slides/TitleSlide'
import Act1Slides from './slides/Act1Slides'
import Act2Slides from './slides/Act2Slides'
import Act3Slides from './slides/Act3Slides'
import Act4Slides from './slides/Act4Slides'
import ClosingSlides from './slides/ClosingSlides'

function App() {
  useEffect(() => {
    const deck = new Reveal({
      plugins: [Highlight],
      hash: true,
      slideNumber: true,
      transition: 'slide',
      backgroundTransition: 'fade',
      center: true,
      width: 1280,
      height: 720,
      margin: 0.04,
    })

    deck.initialize()

    return () => {
      try {
        deck.destroy()
      } catch (e) {
        // ignore
      }
    }
  }, [])

  return (
    <div className="reveal">
      <div className="slides">
        <TitleSlide />
        <Act1Slides />
        <Act2Slides />
        <Act3Slides />
        <Act4Slides />
        <ClosingSlides />
      </div>
    </div>
  )
}

export default App
