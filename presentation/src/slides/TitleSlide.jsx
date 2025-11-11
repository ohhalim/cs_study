export default function TitleSlide() {
  return (
    <section>
      <h1>실무에서 마주친<br/>동시성 문제 해결기</h1>
      <h3>From Bug to Solution</h3>
      <p style={{ marginTop: '60px', fontSize: '1.2em' }}>
        <span className="emoji">🐛</span> 1000명이 동시에 좋아요를 누르면 어떻게 될까?
      </p>
      <p style={{ marginTop: '100px', fontSize: '0.8em', opacity: 0.7 }}>
        CS 스터디 발표<br/>
        {new Date().toLocaleDateString('ko-KR', {
          year: 'numeric',
          month: 'long',
          day: 'numeric'
        })}
      </p>
    </section>
  )
}
