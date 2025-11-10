export default function Act1Slides() {
  return (
    <>
      {/* Act 1: 문제의 발견 */}
      <section>
        <section>
          <h2><span className="emoji">🎬</span> Act 1: 문제의 발견</h2>
          <p style={{ fontSize: '1.5em', marginTop: '60px' }}>
            평범한 시작...
          </p>
        </section>

        <section>
          <h3>커뮤니티 서비스 개발 중...</h3>
          <div className="box">
            <p><strong>기능: 댓글 좋아요 <span className="emoji">👍</span></strong></p>
            <pre style={{ marginTop: '30px' }}><code className="plaintext">{`User → API → Service → Repository → DB`}</code></pre>
          </div>
          <div style={{ marginTop: '40px', textAlign: 'left' }}>
            <p className="highlight-green"><span className="emoji">✅</span> 로컬 테스트: 정상 동작</p>
            <p className="highlight-green"><span className="emoji">✅</span> 단위 테스트: 통과</p>
            <p className="highlight-green"><span className="emoji">✅</span> 배포 완료</p>
          </div>
          <p style={{ marginTop: '40px', fontSize: '1.2em' }}>
            "완벽해 보였습니다..."
          </p>
        </section>

        <section>
          <h3><span className="emoji">🔴</span> 부하 테스트 중 이상한 일이...</h3>
          <div className="comparison">
            <div className="comparison-item">
              <h4>테스트 시나리오</h4>
              <ul style={{ textAlign: 'left', fontSize: '0.9em' }}>
                <li>1000명의 사용자</li>
                <li>동시에 같은 댓글에 좋아요</li>
              </ul>
            </div>
            <div className="comparison-item">
              <h4>예상 결과</h4>
              <pre><code className="plaintext">좋아요 수: 1000 ✅</code></pre>
            </div>
          </div>
          <div className="box" style={{ marginTop: '30px', background: 'rgba(231, 76, 60, 0.2)' }}>
            <h4 className="highlight-red">실제 결과</h4>
            <pre><code className="plaintext">{`좋아요 수: 347 ❌
좋아요 수: 523 ❌
좋아요 수: 681 ❌

매번 다른 숫자!`}</code></pre>
          </div>
          <p style={{ fontSize: '1.5em', marginTop: '30px' }}>"뭐가 문제지?" <span className="emoji">🤔</span></p>
        </section>

        <section>
          <h3>문제의 코드</h3>
          <pre><code className="java language-java" data-line-numbers="2-12|6">{`// ❌ 문제가 있는 코드
@Transactional
public void toggleCommentLike(Long commentId, Long userId) {
    // 1. 댓글 조회
    CommunityCommentEntity comment =
        commentRepository.findById(commentId)
            .orElseThrow();

    // 2. 좋아요 수 증가
    comment.incrementLikeCount();  // ⚠️ 여기!

    // 3. 저장
    commentRepository.save(comment);
}`}</code></pre>
          <p style={{ marginTop: '30px' }}>
            언뜻 보면 문제가 없어 보이는데...<br/>
            <span className="highlight-red">incrementLikeCount()</span>에서 문제 발생!
          </p>
        </section>

        <section>
          <h3>동시 실행 시나리오 <span className="emoji">⏱️</span></h3>
          <div style={{ display: 'flex', gap: '40px', justifyContent: 'center', alignItems: 'flex-start' }}>
            <div className="timeline fragment">
              <h4>Thread 1:</h4>
              <pre style={{ fontSize: '0.8em' }}><code className="plaintext">{`1. count 읽기 (0)  ⬅──┐
2. +1 계산 (1)        │ 동시에!
3. 쓰기 (1)           │`}</code></pre>
            </div>
            <div className="timeline fragment">
              <h4>Thread 2:</h4>
              <pre style={{ fontSize: '0.8em' }}><code className="plaintext">{`1. count 읽기 (0)  ⬅──┘
2. +1 계산 (1)
3. 쓰기 (1)`}</code></pre>
            </div>
          </div>
          <div className="box fragment" style={{ marginTop: '40px', background: 'rgba(231, 76, 60, 0.2)' }}>
            <h4>결과</h4>
            <pre><code className="plaintext">{`예상: 2
실제: 1  ❌`}</code></pre>
          </div>
          <p className="fragment" style={{ fontSize: '1.5em', marginTop: '30px' }}>
            <span className="emoji">⚡</span> <span className="highlight-red">Race Condition</span> 발생!
          </p>
        </section>

        <section>
          <h3>Act 1 정리</h3>
          <div className="box">
            <h4><span className="emoji">🔍</span> 발견한 것</h4>
            <ul style={{ textAlign: 'left' }}>
              <li>단일 사용자 환경에서는 문제 없음</li>
              <li><strong>동시에 여러 사용자</strong>가 접근하면 문제 발생</li>
              <li>데이터 정합성 깨짐</li>
            </ul>
          </div>
          <div className="box" style={{ marginTop: '30px' }}>
            <h4><span className="emoji">🎯</span> 해결해야 할 것</h4>
            <p>여러 스레드가 동시에 같은 데이터를 수정할 때<br/>
            데이터 정합성을 어떻게 보장할 것인가?</p>
          </div>
          <p style={{ fontSize: '1.3em', marginTop: '40px' }}>
            <span className="emoji">💡</span> 키워드: <span className="highlight-blue">동시성 제어</span> (Concurrency Control)
          </p>
        </section>
      </section>
    </>
  )
}
