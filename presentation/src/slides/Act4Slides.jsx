export default function Act4Slides() {
  return (
    <>
      {/* Act 4: 실무의 적용 */}
      <section>
        <section>
          <h2><span className="emoji">💼</span> Act 4: 실무의 적용</h2>
          <p style={{ fontSize: '1.5em', marginTop: '60px' }}>
            실무에서는 어떻게 쓸까?
          </p>
        </section>

        <section>
          <h3>Pattern 1: 계층형 스레드 풀</h3>
          <div className="box">
            <h4>문제 상황</h4>
            <p>모든 작업을 하나의 스레드 풀로? → 중요한 작업이 밀릴 수 있음!</p>
          </div>
          <pre style={{ marginTop: '30px' }}><code className="java language-java" data-line-numbers="5-9|12-16">{`@Configuration
public class ThreadPoolConfig {

    // 1. API 요청용 (빠른 응답 필요)
    @Bean("apiThreadPool")
    public ThreadPoolTaskExecutor apiPool() {
        executor.setCorePoolSize(50);
        executor.setMaxPoolSize(100);
        return executor;
    }

    // 2. 백그라운드 작업용 (느려도 됨)
    @Bean("backgroundThreadPool")
    public ThreadPoolTaskExecutor bgPool() {
        executor.setCorePoolSize(10);
        executor.setMaxPoolSize(20);
        return executor;
    }
}`}</code></pre>
          <p style={{ marginTop: '20px', fontSize: '0.9em' }}>
            <span className="emoji">✅</span> 이메일 발송이 밀려도 사용자 API는 빠르게 처리!
          </p>
        </section>

        <section>
          <h3>Pattern 2: 비동기 처리</h3>
          <div className="comparison">
            <div className="comparison-item" style={{ background: 'rgba(231, 76, 60, 0.2)' }}>
              <h4>❌ 동기 처리</h4>
              <pre><code className="java language-java" style={{ fontSize: '0.6em' }}>{`public Order process(Order order) {
  checkStock(order);      // 300ms
  processPayment(order);  // 500ms
  reserveShipping(order); // 200ms

  return order;  // 총 1000ms
}`}</code></pre>
            </div>
            <div className="comparison-item" style={{ background: 'rgba(39, 174, 96, 0.2)' }}>
              <h4>✅ 비동기 처리</h4>
              <pre><code className="java language-java" style={{ fontSize: '0.6em' }}>{`public CompletableFuture<Order>
  process(Order order) {
  CompletableFuture<Void> stock =
    runAsync(() -> checkStock(order));
  CompletableFuture<Void> payment =
    runAsync(() -> processPayment(order));
  CompletableFuture<Void> shipping =
    runAsync(() -> reserveShipping(order));

  return allOf(stock, payment, shipping)
    .thenApply(v -> order);
  // 총 500ms (2배 빨라짐!)
}`}</code></pre>
            </div>
          </div>
          <p style={{ marginTop: '20px', fontSize: '1.1em' }}>
            <span className="emoji">🚀</span> 독립적인 작업은 비동기로!
          </p>
        </section>

        <section>
          <h3>Pattern 3: 배치 처리</h3>
          <div className="comparison">
            <div className="comparison-item" style={{ background: 'rgba(231, 76, 60, 0.2)' }}>
              <h4>❌ 개별 처리</h4>
              <pre><code className="java language-java" style={{ fontSize: '0.65em' }}>{`// 각 요청마다 DB 저장
for (Task task : tasks) {
  repository.save(task);
  // 1000번 쿼리
}

// 10,000개 → 30초`}</code></pre>
            </div>
            <div className="comparison-item" style={{ background: 'rgba(39, 174, 96, 0.2)' }}>
              <h4>✅ 배치 처리</h4>
              <pre><code className="java language-java" style={{ fontSize: '0.65em' }}>{`// 100개씩 모아서 처리
@Scheduled(fixedRate = 100)
public void processBatch() {
  List<Task> batch =
    queue.drainTo(100);
  repository.saveAll(batch);
  // Bulk Insert
}

// 10,000개 → 2초
// 15배 빨라짐!`}</code></pre>
            </div>
          </div>
          <p style={{ marginTop: '20px', fontSize: '0.9em' }}>
            <span className="emoji">💡</span> 적용 사례: 로그 저장, 알림 발송, 통계 집계
          </p>
        </section>

        <section>
          <h3>성능 모니터링 필수!</h3>
          <pre><code className="java language-java">{`@Scheduled(fixedRate = 5000)
public void checkThreadPool() {
    ThreadPoolExecutor pool = executor.getThreadPoolExecutor();

    int active = pool.getActiveCount();
    int poolSize = pool.getPoolSize();
    int queueSize = pool.getQueue().size();

    // 경고 조건
    if (active == pool.getMaximumPoolSize()) {
        alert("스레드 풀 포화!");
    }

    if (queueSize > queueCapacity * 0.8) {
        alert("큐가 80% 차있음!");
    }
}`}</code></pre>
          <p style={{ marginTop: '20px', fontSize: '1.1em' }}>
            <span className="emoji">📊</span> 도구: Grafana, Spring Boot Actuator, APM
          </p>
        </section>

        <section>
          <h3>문제 상황별 대응</h3>
          <table style={{ fontSize: '0.7em', width: '100%' }}>
            <thead>
              <tr>
                <th>상황</th>
                <th>증상</th>
                <th>해결</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>스레드 풀 포화</strong></td>
                <td>응답 시간 급증</td>
                <td>풀 크기 증가<br/>작업 경량화</td>
              </tr>
              <tr>
                <td><strong>Deadlock</strong></td>
                <td>특정 요청 멈춤</td>
                <td>락 순서 일관성<br/>Timeout 설정</td>
              </tr>
              <tr>
                <td><strong>Race Condition</strong></td>
                <td>숫자가 안 맞음</td>
                <td>적절한 락 사용<br/>테스트 작성</td>
              </tr>
              <tr>
                <td><strong>성능 저하</strong></td>
                <td>전반적으로 느림</td>
                <td>Lock 범위 최소화<br/>Lock-Free 자료구조</td>
              </tr>
            </tbody>
          </table>
        </section>

        <section>
          <h3><span className="emoji">⚠️</span> 주의사항</h3>
          <div className="box">
            <h4>하지 말아야 할 것</h4>
            <ul style={{ textAlign: 'left', fontSize: '0.85em' }}>
              <li><strong>무분별한 synchronized</strong>: 필요한 부분만 동기화</li>
              <li><strong>ThreadLocal 누수</strong>: 반드시 remove() 호출</li>
              <li><strong>스레드 수 무한정 증가</strong>: 제한 설정 필수</li>
            </ul>
          </div>
          <div className="box" style={{ marginTop: '20px', background: 'rgba(39, 174, 96, 0.2)' }}>
            <h4>베스트 프랙티스</h4>
            <ul style={{ textAlign: 'left', fontSize: '0.85em' }}>
              <li><span className="emoji">📊</span> <strong>측정하고 최적화</strong>: 추측 금지!</li>
              <li><span className="emoji">🧪</span> <strong>테스트 작성</strong>: 동시성 테스트 필수</li>
              <li><span className="emoji">📝</span> <strong>문서화</strong>: @ThreadSafe 명시</li>
              <li><span className="emoji">✨</span> <strong>단순하게</strong>: 복잡한 최적화보다 명확한 코드</li>
            </ul>
          </div>
        </section>

        <section>
          <h3>실무 체크리스트 ✅</h3>
          <div className="comparison">
            <div className="comparison-item">
              <h4>설계 & 구현</h4>
              <ul style={{ textAlign: 'left', fontSize: '0.75em' }}>
                <li>□ 공유 자원 파악</li>
                <li>□ 동시성 제어 방법 선택</li>
                <li>□ Lock 범위 최소화</li>
                <li>□ Deadlock 가능성 검토</li>
              </ul>
            </div>
            <div className="comparison-item">
              <h4>테스트 & 운영</h4>
              <ul style={{ textAlign: 'left', fontSize: '0.75em' }}>
                <li>□ 동시성 테스트 작성</li>
                <li>□ 부하 테스트 수행</li>
                <li>□ 모니터링 설정</li>
                <li>□ 문서화 완료</li>
              </ul>
            </div>
          </div>
          <p style={{ fontSize: '1.1em', marginTop: '40px' }}>
            <span className="emoji">🎯</span> 이 체크리스트로 대부분의 동시성 문제 예방 가능!
          </p>
        </section>

        <section>
          <h3>실무 적용 핵심 정리</h3>
          <div className="box">
            <h4><span className="emoji">📌</span> 기본 원칙</h4>
            <pre style={{ fontSize: '0.75em', textAlign: 'left' }}><code className="plaintext">{`1. 측정할 수 없으면 개선할 수 없다
2. 조기 최적화는 악의 근원
3. 단순함이 최고의 최적화
4. 테스트로 검증하라`}</code></pre>
          </div>
          <div className="box" style={{ marginTop: '20px' }}>
            <h4><span className="emoji">🚀</span> 핵심 메시지</h4>
            <ul style={{ textAlign: 'left', fontSize: '0.9em' }}>
              <li><strong>동시성 제어</strong>는 선택이 아닌 필수!</li>
              <li><strong>원리를 이해</strong>하면 응용 가능!</li>
              <li><strong>실무에서는 균형</strong>이 중요!</li>
            </ul>
          </div>
        </section>
      </section>
    </>
  )
}
