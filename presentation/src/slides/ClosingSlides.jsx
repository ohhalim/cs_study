export default function ClosingSlides() {
  return (
    <>
      {/* 마무리 */}
      <section>
        <section>
          <h2><span className="emoji">🎯</span> 우리가 걸어온 길</h2>
          <div style={{ textAlign: 'left', fontSize: '0.9em', lineHeight: '2' }}>
            <p><strong>Act 1: 문제의 발견</strong></p>
            <p style={{ paddingLeft: '30px', color: '#888' }}>
              평범한 코드 → 동시성 문제 발생 → Race Condition 발견
            </p>

            <p style={{ marginTop: '20px' }}><strong>Act 2: 해결의 과정</strong></p>
            <p style={{ paddingLeft: '30px', color: '#888' }}>
              비관적 락 선택 → 코드 적용 → 테스트 → 1000개 정확히 처리 성공!
            </p>

            <p style={{ marginTop: '20px' }}><strong>Act 3: 원리의 이해</strong></p>
            <p style={{ paddingLeft: '30px', color: '#888' }}>
              프로세스 & 스레드 → 동시성 문제 본질 → 락의 원리 → 실무 고려사항
            </p>

            <p style={{ marginTop: '20px' }}><strong>Act 4: 실무의 적용</strong></p>
            <p style={{ paddingLeft: '30px', color: '#888' }}>
              실무 패턴 → 모니터링 → 문제 해결 → 체크리스트 & 베스트 프랙티스
            </p>
          </div>
          <div className="box" style={{ marginTop: '40px', background: 'rgba(66, 185, 131, 0.2)' }}>
            <p style={{ fontSize: '1.2em' }}>
              <span className="emoji">💡</span> 단순한 버그 픽스가 아니라<br/>
              <span className="highlight-green">깊은 이해를 통한 진정한 성장!</span>
            </p>
          </div>
        </section>

        <section>
          <h2><span className="emoji">💬</span> 여러분에게 전하고 싶은 말</h2>
          <div className="box">
            <h3 style={{ color: '#42b983' }}><span className="emoji">💡</span> 실무의 가치</h3>
            <p style={{ fontSize: '1.1em', lineHeight: '1.8' }}>
              이론은 책으로 배울 수 있지만,<br/>
              실무 경험은 직접 부딪혀야 얻을 수 있습니다.
            </p>
          </div>
          <div className="box" style={{ marginTop: '40px' }}>
            <h3 style={{ color: '#42b983' }}><span className="emoji">🚀</span> 계속 배우기</h3>
            <p style={{ fontSize: '0.9em', lineHeight: '1.8', textAlign: 'left' }}>
              오늘 배운 것은 시작일 뿐!<br/><br/>
              <strong>책:</strong> "Java Concurrency in Practice"<br/>
              <strong>공식 문서:</strong> Java Tutorials - Concurrency<br/>
              <strong>실습:</strong> 직접 동시성 버그 만들고 해결하기
            </p>
          </div>
        </section>

        <section>
          <h2><span className="emoji">🎯</span> 핵심 메시지</h2>
          <div className="box" style={{
            background: 'rgba(66, 185, 131, 0.2)',
            padding: '40px',
            fontSize: '1.3em',
            lineHeight: '2'
          }}>
            <p style={{ fontSize: '1.5em' }}><span className="emoji">💎</span></p>
            <p>
              "문제를 만나는 것은 <span className="highlight-green">기회</span>다.<br/>
              그 문제를 <span className="highlight-blue">깊이 이해</span>하면,<br/>
              더 <span className="highlight-green">나은 개발자</span>가 된다."
            </p>
          </div>
          <p style={{ fontSize: '1.5em', marginTop: '60px' }}>
            <span className="emoji">💬</span> 질문 있으신가요?
          </p>
        </section>

        <section data-background-color="#1a1a1a">
          <h1 style={{ fontSize: '2.5em' }}>감사합니다! <span className="emoji">🙏</span></h1>
          <div style={{ marginTop: '80px', fontSize: '1.1em' }}>
            <p style={{ marginBottom: '30px' }}>
              <span className="emoji">📧</span> Email: your-email@example.com
            </p>
            <p style={{ marginBottom: '30px' }}>
              <span className="emoji">💻</span> GitHub: github.com/yourname
            </p>
            <p>
              <span className="emoji">📝</span> 발표 자료: github.com/yourname/presentation
            </p>
          </div>
          <div style={{ marginTop: '80px', fontSize: '0.9em', opacity: 0.7 }}>
            <p><span className="emoji">📚</span> 참고 자료</p>
            <p style={{ fontSize: '0.8em', marginTop: '20px' }}>
              "Java Concurrency in Practice" - Brian Goetz<br/>
              "Operating System Concepts" - Silberschatz<br/>
              Oracle Java Tutorials - Concurrency
            </p>
          </div>
          <p style={{ fontSize: '1.3em', marginTop: '60px' }}>
            <span className="emoji">🚀</span> 함께 성장하는 개발자가 되길 바랍니다!
          </p>
        </section>
      </section>
    </>
  )
}
