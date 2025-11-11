export default function Act3Slides() {
  return (
    <>
      {/* Act 3: 원리의 이해 */}
      <section>
        <section>
          <h2><span className="emoji">🧠</span> Act 3: 원리의 이해</h2>
          <p style={{ fontSize: '1.5em', marginTop: '60px' }}>
            왜 이렇게 해결될까?
          </p>
        </section>

        <section>
          <h3><span className="emoji">🤔</span> 궁금한 점들</h3>
          <div className="box">
            <ul style={{ textAlign: 'left', fontSize: '1.1em', lineHeight: '1.8' }}>
              <li>"동시에 실행된다"는 게 정확히 뭘까?</li>
              <li>스레드가 뭐길래 이런 문제가 생기지?</li>
              <li>왜 락을 걸면 해결될까?</li>
            </ul>
          </div>
          <p style={{ fontSize: '1.2em', marginTop: '50px' }}>
            <span className="emoji">🎯</span> 목표: 표면적 해결이 아닌<br/>
            <span className="highlight-blue">깊은 이해</span>를 통한 진짜 해결!
          </p>
        </section>

        <section>
          <h3>프로세스 (Process)</h3>
          <div className="box">
            <h4>정의</h4>
            <ul style={{ textAlign: 'left', fontSize: '0.9em' }}>
              <li>실행 중인 프로그램</li>
              <li>OS가 자원을 할당하는 기본 단위</li>
            </ul>
          </div>
          <pre style={{ marginTop: '30px' }}><code className="plaintext">{`┌─────────────────────┐
│    Process A        │
├─────────────────────┤
│  Code  (프로그램)    │
│  Data  (전역변수)    │
│  Heap  (동적 할당)   │
│  Stack (함수 호출)   │
└─────────────────────┘
    ↕ 독립적!
┌─────────────────────┐
│    Process B        │
└─────────────────────┘`}</code></pre>
          <div className="box" style={{ marginTop: '30px' }}>
            <p>✅ 독립된 메모리 공간 ✅ 다른 프로세스 접근 불가 ❌ 생성/전환 비용 높음</p>
          </div>
        </section>

        <section>
          <h3>스레드 (Thread)</h3>
          <div className="box">
            <h4>정의</h4>
            <ul style={{ textAlign: 'left', fontSize: '0.9em' }}>
              <li>프로세스 내의 실행 단위</li>
              <li>CPU가 실행하는 가장 작은 단위</li>
            </ul>
          </div>
          <pre style={{ marginTop: '30px' }}><code className="plaintext">{`┌─────────────────────────────┐
│       Process               │
├─────────────────────────────┤
│  Code  (공유) 📖            │
│  Data  (공유) 📊            │
│  Heap  (공유) 📦 ⚠️ 여기!   │
├─────────────────────────────┤
│  Thread 1   Thread 2        │
│  Stack      Stack           │
│  (개인방)    (개인방)         │
└─────────────────────────────┘`}</code></pre>
          <div className="box" style={{ marginTop: '30px', background: 'rgba(231, 76, 60, 0.2)' }}>
            <p>✅ 같은 프로세스 내 자원 공유 ✅ 생성/전환 비용 낮음 <br/>
            <span className="highlight-red">⚠️ 공유 → 동시성 문제!</span></p>
          </div>
        </section>

        <section>
          <h3>왜 스레드를 쓸까?</h3>
          <div className="comparison">
            <div className="comparison-item" style={{ background: 'rgba(231, 76, 60, 0.2)' }}>
              <h4>❌ 스레드 1개</h4>
              <pre><code className="plaintext">{`작업1 → 작업2 → 작업3

한 번에 하나씩만`}</code></pre>
            </div>
            <div className="comparison-item" style={{ background: 'rgba(39, 174, 96, 0.2)' }}>
              <h4>✅ 스레드 N개</h4>
              <pre><code className="plaintext">{`작업1 ↘
작업2 → 동시 처리!
작업3 ↗`}</code></pre>
            </div>
          </div>
          <div className="box" style={{ marginTop: '40px' }}>
            <h4>실제 예: 웹 서버 (Tomcat)</h4>
            <pre><code className="plaintext" style={{ fontSize: '0.7em' }}>{`Thread 1: User A 요청 처리
Thread 2: User B 요청 처리
Thread 3: User C 요청 처리
...
Thread 200: User 200 요청 처리

➡️ 200명을 동시에 처리!`}</code></pre>
          </div>
          <p style={{ fontSize: '1.3em', marginTop: '30px' }}>
            <span className="emoji">🚀</span> <span className="highlight-green">효율성 극대화!</span>
          </p>
        </section>

        <section>
          <h3>Java Thread = OS Thread</h3>
          <pre><code className="plaintext">{`Java Application
  Thread t = new Thread()
        ↓ JNI
      JVM
        ↓ System Call
    OS Kernel
  Native Thread 생성
        ↓
  1:1 매핑`}</code></pre>
          <div className="box" style={{ marginTop: '30px' }}>
            <h4>비용</h4>
            <table style={{ fontSize: '0.8em', width: '100%' }}>
              <tbody>
                <tr>
                  <td>생성</td>
                  <td>~1ms</td>
                </tr>
                <tr>
                  <td>메모리</td>
                  <td>1MB per thread</td>
                </tr>
                <tr>
                  <td>컨텍스트 스위칭</td>
                  <td>5-10μs</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p style={{ marginTop: '30px', fontSize: '1.1em' }}>
            <span className="emoji">💡</span> Java 스레드 1개 = OS 스레드 1개<br/>
            → 스레드 생성은 비싸다! → <span className="highlight-blue">스레드 풀 사용!</span>
          </p>
        </section>

        <section>
          <h3>동시성 문제의 본질</h3>
          <div className="box">
            <h4><span className="emoji">✅</span> 독립 자원 (문제 없음)</h4>
            <pre><code className="plaintext">{`Thread 1: Stack 1 ✅
Thread 2: Stack 2 ✅
→ 각자의 공간, 간섭 없음`}</code></pre>
          </div>
          <div className="box" style={{ marginTop: '20px', background: 'rgba(231, 76, 60, 0.2)' }}>
            <h4><span className="emoji">⚠️</span> 공유 자원 (문제 발생!)</h4>
            <pre><code className="plaintext">{`Thread 1 ↘
         Heap (count 변수) ⚠️
Thread 2 ↗
→ 동시 접근, Race Condition!`}</code></pre>
          </div>
          <p style={{ fontSize: '1.2em', marginTop: '30px' }}>
            핵심: <span className="highlight-red">공유 자원 경쟁</span>
          </p>
        </section>

        <section>
          <h3>두 가지 문제</h3>
          <div className="comparison">
            <div className="comparison-item">
              <h4>1. 원자성 (Atomicity)</h4>
              <pre><code className="java language-java">{`count++ 는 3단계:
1. READ  count
2. ADD   +1
3. WRITE count

→ 중간에 끊기면 문제!`}</code></pre>
            </div>
            <div className="comparison-item">
              <h4>2. 가시성 (Visibility)</h4>
              <pre><code className="java language-java">{`Thread 1: count = 1
          (CPU 캐시)
Thread 2: count = 0
          (아직 안 보임)

→ 최신값이 안 보임!`}</code></pre>
            </div>
          </div>
          <p style={{ fontSize: '1.2em', marginTop: '30px' }}>
            <span className="emoji">🔑</span> 락이 이 두 문제를 모두 해결!
          </p>
        </section>

        <section>
          <h3>락(Lock)의 원리</h3>
          <div className="box">
            <h4><span className="emoji">🚪</span> 락 = "화장실 사용 중" 표지판</h4>
            <pre style={{ marginTop: '20px' }}><code className="plaintext">{`Thread 1: [락 획득] → 작업 중... → [락 해제]
Thread 2:            [대기...]       [락 획득]`}</code></pre>
          </div>
          <div className="comparison" style={{ marginTop: '30px' }}>
            <div className="comparison-item">
              <h4>원자성 보장</h4>
              <pre><code className="plaintext">{`락 획득
  ↓
READ + ADD + WRITE
(중간에 안 끊김!)
  ↓
락 해제`}</code></pre>
            </div>
            <div className="comparison-item">
              <h4>가시성 보장</h4>
              <pre><code className="plaintext">{`트랜잭션 커밋 시
  ↓
변경사항 DB 반영
  ↓
다른 스레드가
최신값 읽기 가능`}</code></pre>
            </div>
          </div>
        </section>

        <section>
          <h3>다양한 동시성 제어 도구</h3>
          <div className="comparison">
            <div className="comparison-item">
              <h4>Application Level</h4>
              <pre><code className="java language-java" style={{ fontSize: '0.6em' }}>{`// synchronized
synchronized void method() {
  count++;
}

// AtomicInteger (Lock-Free)
AtomicInteger count;
count.incrementAndGet();

// volatile (가시성만)
volatile boolean flag;`}</code></pre>
            </div>
            <div className="comparison-item">
              <h4>Database Level</h4>
              <pre><code className="sql language-sql" style={{ fontSize: '0.6em' }}>{`-- 비관적 락
-- (충돌 많을 때)
SELECT ... FOR UPDATE


-- 낙관적 락
-- (충돌 적을 때)
@Version Long version;`}</code></pre>
            </div>
          </div>
          <p style={{ fontSize: '1.1em', marginTop: '30px' }}>
            <span className="emoji">🎯</span> 선택 기준: <strong>작업 유형</strong> + <strong>충돌 빈도</strong>
          </p>
        </section>

        <section>
          <h3>스레드 풀의 필요성</h3>
          <div className="comparison">
            <div className="comparison-item" style={{ background: 'rgba(231, 76, 60, 0.2)' }}>
              <h4>❌ 매번 생성</h4>
              <pre><code className="java language-java" style={{ fontSize: '0.65em' }}>{`for (int i = 0; i < 1000; i++) {
  new Thread(() -> task())
    .start();
}

// 1000개 생성: ~1초
// 메모리: 1GB
// 컨텍스트 스위칭 폭증`}</code></pre>
            </div>
            <div className="comparison-item" style={{ background: 'rgba(39, 174, 96, 0.2)' }}>
              <h4>✅ 스레드 풀</h4>
              <pre><code className="java language-java" style={{ fontSize: '0.65em' }}>{`ExecutorService pool =
  Executors.newFixedThreadPool(20);

for (int i = 0; i < 1000; i++) {
  pool.submit(() -> task());
}

// 20개만 생성, 재사용
// 메모리: 20MB
// 효율적!`}</code></pre>
            </div>
          </div>
        </section>

        <section>
          <h3>CPU-Bound vs I/O-Bound</h3>
          <div className="comparison">
            <div className="comparison-item">
              <h4>CPU-Bound (계산 많음)</h4>
              <pre><code className="plaintext" style={{ fontSize: '0.7em' }}>{`예: 이미지 처리, 암호화

최적 스레드 수:
코어 수 + 1

4코어 → 5개 스레드

이유:
컨텍스트 스위칭 최소화`}</code></pre>
            </div>
            <div className="comparison-item">
              <h4>I/O-Bound (대기 많음)</h4>
              <pre><code className="plaintext" style={{ fontSize: '0.7em' }}>{`예: DB 쿼리, HTTP 요청

최적 스레드 수:
코어 수 × 10 ~ 100

4코어 → 40~400개

이유:
대기 시간 활용`}</code></pre>
            </div>
          </div>
          <div className="box" style={{ marginTop: '30px' }}>
            <p><span className="emoji">💡</span> 우리 서비스 (댓글 좋아요): DB 쿼리 위주 → <strong>I/O-Bound</strong></p>
          </div>
        </section>

        <section>
          <h3>Act 3 정리</h3>
          <div className="box">
            <h4><span className="emoji">🧠</span> 배운 것들</h4>
            <ul style={{ textAlign: 'left', fontSize: '0.9em' }}>
              <li><strong>프로세스 & 스레드</strong>: 독립 vs 공유</li>
              <li><strong>동시성 문제</strong>: 원자성 & 가시성</li>
              <li><strong>락의 원리</strong>: 순차 처리로 해결</li>
              <li><strong>스레드 풀</strong>: 생성 비용 절약</li>
              <li><strong>작업 유형</strong>: CPU-Bound vs I/O-Bound</li>
            </ul>
          </div>
          <p style={{ fontSize: '1.3em', marginTop: '40px' }}>
            <span className="emoji">🎯</span> 핵심: 원리를 이해하면<br/>
            <span className="highlight-blue">다양한 상황에 대응 가능!</span>
          </p>
        </section>
      </section>
    </>
  )
}
