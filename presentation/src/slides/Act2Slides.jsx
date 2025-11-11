export default function Act2Slides() {
  return (
    <>
      {/* Act 2: 해결의 과정 */}
      <section>
        <section>
          <h2><span className="emoji">🔧</span> Act 2: 해결의 과정</h2>
          <p style={{ fontSize: '1.5em', marginTop: '60px' }}>
            어떻게 해결할까?
          </p>
        </section>

        <section>
          <h3>동시성 제어 방법 탐색</h3>
          <div className="comparison">
            <div className="comparison-item">
              <h4>1️⃣ Application Level</h4>
              <pre><code className="java language-java">{`synchronized void increment() {
    count++;
}

AtomicInteger count;

Lock lock = new ReentrantLock();`}</code></pre>
            </div>
            <div className="comparison-item">
              <h4>2️⃣ Database Level</h4>
              <pre><code className="sql language-sql">{`-- 비관적 락
SELECT ... FOR UPDATE

-- 낙관적 락
@Version column`}</code></pre>
            </div>
          </div>
          <p style={{ fontSize: '1.2em', marginTop: '40px' }}>
            <span className="emoji">🤔</span> 어떤 걸 선택해야 할까?
          </p>
        </section>

        <section>
          <h3>왜 비관적 락을 선택했나?</h3>
          <div className="comparison">
            <div className="comparison-item">
              <h4>낙관적 락</h4>
              <ul style={{ textAlign: 'left', fontSize: '0.85em' }}>
                <li>가정: "충돌 안 날 거야"</li>
                <li>방식: Version 체크</li>
                <li>실패 시: 재시도 필요</li>
              </ul>
              <p className="highlight-red" style={{ marginTop: '20px' }}>
                👎 좋아요는 충돌 자주 발생<br/>
                → 재시도 많음
              </p>
            </div>
            <div className="comparison-item">
              <h4>비관적 락</h4>
              <ul style={{ textAlign: 'left', fontSize: '0.85em' }}>
                <li>가정: "충돌 날 거야"</li>
                <li>방식: 미리 락 걸기</li>
                <li>실패 시: 대기 후 처리</li>
              </ul>
              <p className="highlight-green" style={{ marginTop: '20px' }}>
                👍 충돌 많은 경우<br/>
                안정적!
              </p>
            </div>
          </div>
          <div className="box" style={{ marginTop: '30px' }}>
            <h4>우리 상황</h4>
            <p>✅ 인기 댓글은 좋아요가 몰림 ✅ 충돌 빈번 ✅ 데이터 정합성 최우선</p>
            <p style={{ fontSize: '1.3em', marginTop: '20px' }}>
              <span className="emoji">➡️</span> <span className="highlight-green">비관적 락 선택!</span>
            </p>
          </div>
        </section>

        <section>
          <h3>해결 코드 - Repository</h3>
          <pre><code className="java language-java" data-line-numbers="|6-7">{`// Repository에 비관적 락 추가
@Repository
public interface CommunityCommentRepository
        extends JpaRepository<CommunityCommentEntity, Long> {

    // ✅ 비관적 락 쿼리 추가
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT c FROM CommunityCommentEntity c " +
           "WHERE c.commentId = :commentId")
    Optional<CommunityCommentEntity> findByIdWithPessimisticLock(
        @Param("commentId") Long commentId
    );
}`}</code></pre>
          <p style={{ marginTop: '30px' }}>
            <span className="highlight-blue">@Lock</span> 어노테이션 하나로 해결!
          </p>
        </section>

        <section>
          <h3>SQL로는 어떻게 번역될까?</h3>
          <pre><code className="sql language-sql">{`-- JPA가 자동으로 생성하는 SQL

SELECT *
FROM community_comment
WHERE comment_id = ?
FOR UPDATE;  -- ⬅️ 이게 핵심!`}</code></pre>
          <div className="box" style={{ marginTop: '40px' }}>
            <h4><span className="emoji">🔑</span> FOR UPDATE의 의미</h4>
            <ul style={{ textAlign: 'left', fontSize: '0.9em' }}>
              <li>이 행(row)에 <strong>배타적 락</strong> 설정</li>
              <li>트랜잭션이 끝날 때까지 다른 트랜잭션은 <strong>대기</strong></li>
              <li>"내가 쓸 테니까 다른 사람은 기다려!"</li>
            </ul>
          </div>
        </section>

        <section>
          <h3>해결 코드 - Service</h3>
          <pre><code className="java language-java" data-line-numbers="5-7">{`// ✅ 비관적 락 적용
@Service
@Transactional
public class CommentLikeService {

    public void toggleCommentLike(Long commentId, Long userId) {
        // 1. 비관적 락으로 댓글 조회
        CommunityCommentEntity comment =
            commentRepository.findByIdWithPessimisticLock(commentId)
                .orElseThrow();

        // 2. 좋아요 토글
        if (existsLike(comment, user)) {
            comment.decrementLikeCount();  // 좋아요 취소
            deleteLike(comment, user);
        } else {
            comment.incrementLikeCount();  // 좋아요 추가
            saveLike(comment, user);
        }

        // 3. 트랜잭션 커밋 시 자동 저장 & 락 해제
    }
}`}</code></pre>
        </section>

        <section>
          <h3>비관적 락 동작 흐름</h3>
          <div style={{ display: 'flex', gap: '30px', justifyContent: 'center', fontSize: '0.8em' }}>
            <div className="timeline">
              <h4 style={{ color: '#42b983' }}>Thread 1:</h4>
              <pre><code className="plaintext">{`10:00:00.000  락 획득 ✅
10:00:00.000  count = 0 읽기
10:00:00.050  count = 1 쓰기
10:00:00.100  커밋
10:00:00.100  락 해제 🔓`}</code></pre>
            </div>
            <div className="timeline">
              <h4 style={{ color: '#e74c3c' }}>Thread 2:</h4>
              <pre><code className="plaintext">{`10:00:00.010  락 대기... ⏳
10:00:00.010  대기 중...
10:00:00.100  락 획득 ✅
10:00:00.100  count = 1 읽기
10:00:00.150  count = 2 쓰기
10:00:00.200  커밋
10:00:00.200  락 해제 🔓`}</code></pre>
            </div>
          </div>
          <div className="box" style={{ marginTop: '30px', background: 'rgba(39, 174, 96, 0.2)' }}>
            <h4 className="highlight-green">결과</h4>
            <pre><code className="plaintext">count = 2 ✅ 정확!</code></pre>
          </div>
          <p style={{ fontSize: '1.2em', marginTop: '20px' }}>
            <span className="emoji">🎯</span> 순차 처리로 정확성 보장!
          </p>
        </section>

        <section>
          <h3>동시성 테스트 작성</h3>
          <pre><code className="java language-java" data-line-numbers="2-5|7-13|16">{`@Test
void concurrencyTest() throws InterruptedException {
    // Given: 1000개의 동시 요청
    int threadCount = 1000;
    ExecutorService executor = Executors.newFixedThreadPool(32);
    CountDownLatch latch = new CountDownLatch(threadCount);

    // When: 동시에 좋아요 클릭
    for (int i = 0; i < threadCount; i++) {
        executor.submit(() -> {
            service.toggleCommentLike(commentId, userId);
            latch.countDown();
        });
    }
    latch.await();

    // Then: 정확히 1000개
    assertThat(comment.getLikeCount()).isEqualTo(1000);
}`}</code></pre>
        </section>

        <section>
          <h3>테스트 결과 <span className="emoji">✅</span></h3>
          <div className="comparison">
            <div className="comparison-item" style={{ background: 'rgba(231, 76, 60, 0.2)' }}>
              <h4>Before (락 없음)</h4>
              <pre><code className="plaintext">{`347 ❌
523 ❌
681 ❌
...
매번 다른 숫자`}</code></pre>
            </div>
            <div className="comparison-item" style={{ background: 'rgba(39, 174, 96, 0.2)' }}>
              <h4>After (비관적 락)</h4>
              <pre><code className="plaintext">{`1000 ✅
1000 ✅
1000 ✅
...
항상 정확!`}</code></pre>
            </div>
          </div>
          <p style={{ fontSize: '1.8em', marginTop: '50px' }}>
            <span className="emoji">🎉</span> <span className="highlight-green">문제 해결 성공!</span>
          </p>
        </section>
      </section>
    </>
  )
}
