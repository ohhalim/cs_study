# 동시성 문제 해결기: 문제에서 원리까지
## From Problem to Deep Understanding

> **발표 컨셉**: 문제 발견 → 해결 → 원리 파고들기 → 응용
> **발표 시간**: 40분
> **타겟**: CS 학습을 통한 깊은 이해

---

## 🎬 발표 구조 Overview

```
Act 1: 문제의 발견 (7분)
    ↓
Act 2: 해결의 과정 (7분)
    ↓
Act 3: 원리를 파고들기 (18분) ← 핵심! CS 깊은 내용
    ↓
Act 4: 이 지식으로 할 수 있는 것들 (8분)
    ↓
마무리 (2분)
```

**핵심 스토리:**
```
"코드에 버그 발견 → 비관적 락으로 해결 → 근데 왜 되는 거지?
→ 락이 뭐지? → 스레드가 뭐지? → 프로세스는? → 더 깊이...
→ 이제 이 지식으로 다른 문제도 해결할 수 있다!"
```

---

# Act 1: 문제의 발견 (7분, 6슬라이드)

## 슬라이드 1: 타이틀
```markdown
동시성 문제 해결기
From Bug to Deep Understanding

부제: 1000명이 동시에 좋아요를 누르면?

[이름]
[날짜]
```

**스피커 노트:**
- 단순한 버그 픽스 이야기가 아님
- 문제를 깊이 파고들어 원리를 이해하는 과정

---

## 슬라이드 2: 평범한 시작
```markdown
# 커뮤니티 서비스 - 댓글 좋아요 기능

## 코드
```java
@Transactional
public void toggleLike(Long commentId, Long userId) {
    CommunityComment comment = repository.findById(commentId)
        .orElseThrow();

    comment.incrementLikeCount();  // count++
    repository.save(comment);
}
```

✅ 로컬 테스트: 정상
✅ 단위 테스트: 통과
✅ 배포 완료

"문제없어 보였다..."
```

---

## 슬라이드 3: 버그 발견
```markdown
# 🔴 부하 테스트 결과

## 시나리오
- 1000명의 사용자
- 동시에 같은 댓글에 좋아요

## 예상
```
좋아요 수: 1000
```

## 실제
```
좋아요 수: 347  ❌
좋아요 수: 523  ❌
좋아요 수: 681  ❌

매번 다른 숫자!
```

❓ "뭐가 문제지?"
```

---

## 슬라이드 4: 문제의 시각화
```markdown
# Race Condition 발생!

## Thread 1 실행:
```
시간 0ms:  count = 0 읽기
시간 1ms:  count + 1 = 1 계산
시간 2ms:  count = 1 쓰기
```

## Thread 2 실행:
```
시간 0.5ms: count = 0 읽기  ← 아직 0!
시간 1.5ms: count + 1 = 1 계산
시간 2.5ms: count = 1 쓰기
```

## 결과
```
예상: 2
실제: 1  ❌
```

**핵심:** 둘 다 0을 읽어서 둘 다 1을 씀!
```

---

## 슬라이드 5: count++ 의 비밀
```markdown
# count++ 는 사실 3단계!

## Java 코드
```java
count++
```

## 실제 동작 (바이트코드)
```
1. LOAD  count     // 메모리에서 읽기
2. ADD   1         // +1 계산
3. STORE count     // 메모리에 쓰기
```

## 문제
```
중간에 끊기면 문제 발생!
→ 원자성(Atomicity) 문제
```
```

---

## 슬라이드 6: Act 1 정리
```markdown
# 문제 정리

## 발견한 것
1. 단일 사용자는 문제 없음
2. **동시에 여러 요청**이 들어오면 데이터 깨짐
3. count++ 같은 단순한 연산도 위험

## 원인
- Race Condition (경쟁 상태)
- 원자성 문제

## 해결 필요
"동시성 제어" (Concurrency Control)
```

---

# Act 2: 해결의 과정 (7분, 7슬라이드)

## 슬라이드 7: 해결 방법 탐색
```markdown
# 동시성 제어 방법

## Application Level
```java
synchronized void increment() { }
AtomicInteger count;
ReentrantLock lock;
```

## Database Level
```java
// 비관적 락 (Pessimistic Lock)
SELECT ... FOR UPDATE

// 낙관적 락 (Optimistic Lock)
@Version Long version;
```

## 우리의 선택
DB 데이터 정합성 → **Database Level 락**
```

---

## 슬라이드 8: 비관적 vs 낙관적
```markdown
# 어떤 락을 선택할까?

## 낙관적 락 (Optimistic)
```
가정: "충돌 안 날 거야"
동작: 커밋할 때 version 체크
충돌 시: 재시도 필요
```
👍 충돌이 적을 때 효율적
👎 충돌 많으면 재시도 폭증

## 비관적 락 (Pessimistic)
```
가정: "충돌 날 거야"
동작: 미리 락을 걸어버림
충돌 시: 대기 후 처리
```
👍 충돌 많을 때 안정적
👎 대기 시간 발생

## 우리 상황
```
인기 댓글 → 좋아요 몰림 → 충돌 빈번
→ 비관적 락 선택!
```
```

---

## 슬라이드 9: 해결 코드 - Repository
```java
@Repository
public interface CommunityCommentRepository
        extends JpaRepository<CommunityCommentEntity, Long> {

    // ✅ 비관적 락 추가
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT c FROM CommunityCommentEntity c " +
           "WHERE c.commentId = :commentId")
    Optional<CommunityCommentEntity> findByIdWithPessimisticLock(
        @Param("commentId") Long commentId
    );
}
```

**하이라이트:**
- `@Lock(LockModeType.PESSIMISTIC_WRITE)` 어노테이션 하나로 해결!

---

## 슬라이드 10: SQL 변환
```sql
-- JPA가 생성하는 SQL

SELECT *
FROM community_comment
WHERE comment_id = ?
FOR UPDATE;  -- ← 핵심!
```

**FOR UPDATE의 의미:**
```
- 이 행(row)에 배타적 락 설정
- 트랜잭션 종료까지 다른 트랜잭션은 대기
- "내가 쓸 거니까 기다려!"
```

---

## 슬라이드 11: 해결 코드 - Service
```java
@Service
@Transactional
public class CommentLikeService {

    public void toggleLike(Long commentId, Long userId) {
        // 1. 비관적 락으로 조회
        CommunityComment comment =
            repository.findByIdWithPessimisticLock(commentId)
                .orElseThrow();

        // 2. 좋아요 토글
        if (existsLike(comment, user)) {
            comment.decrementLikeCount();
            deleteLike(comment, user);
        } else {
            comment.incrementLikeCount();
            saveLike(comment, user);
        }

        // 3. 트랜잭션 커밋 시 자동 저장 & 락 해제
    }
}
```

---

## 슬라이드 12: 동작 흐름
```markdown
# 비관적 락 동작

## Thread 1:
```
10:00:00.000  락 획득 ✅
10:00:00.000  count = 0 읽기
10:00:00.050  count = 1 쓰기
10:00:00.100  커밋 & 락 해제 🔓
```

## Thread 2:
```
10:00:00.010  락 획득 시도... ⏳
10:00:00.010  대기 중...
10:00:00.100  락 획득 ✅
10:00:00.100  count = 1 읽기 (최신값!)
10:00:00.150  count = 2 쓰기
10:00:00.200  커밋 & 락 해제 🔓
```

## 결과
```
count = 2 ✅ 정확!
```

**핵심:** 순차 처리로 변환!
```

---

## 슬라이드 13: 테스트 결과
```markdown
# 동시성 테스트

```java
@Test
void concurrencyTest() throws InterruptedException {
    int threadCount = 1000;
    ExecutorService executor = Executors.newFixedThreadPool(32);
    CountDownLatch latch = new CountDownLatch(threadCount);

    for (int i = 0; i < threadCount; i++) {
        executor.submit(() -> {
            service.toggleLike(commentId, userId);
            latch.countDown();
        });
    }
    latch.await();

    assertThat(comment.getLikeCount()).isEqualTo(1000);
}
```

## 결과
```
Before: 347, 523, 681... ❌
After:  1000, 1000, 1000... ✅

성공!
```
```

---

# Act 3: 원리를 파고들기 (18분, 15슬라이드)

> **핵심 파트!** CS 깊은 내용

## 슬라이드 14: 원리 파트 시작
```markdown
# 🤔 그런데...

## 해결은 했는데 궁금한 게 생김
1. 왜 락을 걸면 해결되는 거지?
2. "동시에 실행"이 정확히 뭐지?
3. 스레드가 뭐길래 이런 문제가?
4. 프로세스와는 뭐가 다른데?

## 이제부터
```
표면적 해결이 아닌
깊은 이해를 위한 여정!
```

## 목표
```
원리를 이해하면
→ 다른 문제도 해결 가능
→ 더 나은 선택 가능
```
```

---

## 슬라이드 15: 락의 원리 - 원자성
```markdown
# 락이 해결하는 것 ① 원자성

## 문제: count++ 는 3단계
```java
1. LOAD  count
2. ADD   1
3. STORE count

→ 중간에 끊기면 문제!
```

## 해결: 락으로 묶기
```java
[락 획득]
  1. LOAD  count
  2. ADD   1
  3. STORE count
[락 해제]

→ 중간에 안 끊김!
→ 원자성 보장!
```

**비유:** 화장실 문 잠그기
- 문 잠그면 → 끝날 때까지 아무도 못 들어옴
- 락 걸면 → 끝날 때까지 다른 스레드 못 들어옴
```

---

## 슬라이드 16: 락의 원리 - 가시성
```markdown
# 락이 해결하는 것 ② 가시성

## 문제: CPU 캐시
```
Thread 1: count = 1 (CPU1 캐시에만 존재)
Thread 2: count = 0 (아직 안 보임!)

→ 최신값이 안 보임!
→ Visibility 문제
```

## 해결: 락 & 트랜잭션
```
Thread 1:
  [락 획득]
  count = 1로 변경
  [트랜잭션 커밋] → DB에 반영 (Main Memory)
  [락 해제]

Thread 2:
  [락 획득]
  DB에서 읽기 → count = 1 (최신값!)
```

**핵심:**
- DB 커밋 = Main Memory에 쓰기
- 다른 스레드는 Main Memory에서 읽기
- → 가시성 보장!
```

---

## 슬라이드 17: 그럼 스레드가 뭔데?
```markdown
# 스레드(Thread)란?

## 정의
- CPU가 실행하는 가장 작은 단위
- 프로세스 내의 실행 흐름

## 왜 필요한가?
```
한 번에 한 가지만:
  작업1 → 작업2 → 작업3 (순차)

여러 개 동시에:
  작업1 ↘
  작업2 → 동시 처리!
  작업3 ↗
```

## 예시: 웹 서버 (Tomcat)
```
Thread 1: 사용자 A의 요청 처리
Thread 2: 사용자 B의 요청 처리
Thread 3: 사용자 C의 요청 처리
...
Thread 200: 사용자 200의 요청 처리

→ 200명을 동시에 처리!
```
```

---

## 슬라이드 18: 프로세스 vs 스레드
```markdown
# 프로세스 vs 스레드

## 프로세스 (Process)
```
┌─────────────────────┐
│    Process A        │
├─────────────────────┤
│  Code               │
│  Data               │
│  Heap               │
│  Stack              │
└─────────────────────┘
    ↕ 독립적!
┌─────────────────────┐
│    Process B        │
└─────────────────────┘
```
- 독립된 메모리 공간
- 다른 프로세스 접근 불가
- 생성 비용 높음 (무거움)

## 스레드 (Thread)
```
┌─────────────────────────────┐
│       Process               │
├─────────────────────────────┤
│  Code  (공유) 📖            │
│  Data  (공유) 📊            │
│  Heap  (공유) 📦 ⚠️         │
├─────────────────────────────┤
│  Thread 1   Thread 2        │
│  Stack      Stack           │
│  (독립)      (독립)          │
└─────────────────────────────┘
```
- 메모리 공유 (Code, Data, Heap)
- Stack만 독립적
- 생성 비용 낮음 (가벼움)

## 핵심 차이
```
프로세스: 독립 → 안전하지만 무거움
스레드:   공유 → 가볍지만 동시성 문제!
```
```

---

## 슬라이드 19: 메모리 구조와 동시성 문제
```markdown
# 왜 동시성 문제가 생기는가?

## Stack (독립) - 문제 없음
```
Thread 1: Stack 1 (독립적)
  - 지역변수 a
  - 지역변수 b

Thread 2: Stack 2 (독립적)
  - 지역변수 c
  - 지역변수 d

→ 서로 간섭 없음 ✅
```

## Heap (공유) - 문제 발생!
```
Thread 1 ↘
         Heap (count 변수) ⚠️
Thread 2 ↗

→ 동시 접근, Race Condition! ❌
```

## 결론
```
공유 자원 (Heap) + 동시 접근
= 동시성 문제의 본질!
```
```

---

## 슬라이드 20: Java Thread의 비밀
```markdown
# Java Thread = OS Thread

## 구조
```
Java 코드:
  new Thread().start()
        ↓
      JVM
        ↓ JNI (Java Native Interface)
   OS Kernel
        ↓ System Call (pthread_create)
  Native Thread 생성
        ↓
   1:1 매핑!
```

## 의미
```
Java 스레드 1개 = OS 스레드 1개
→ 진짜 스레드!
```

## 비용
```
생성 시간:    ~1ms
메모리:       ~1MB (Stack 크기)
컨텍스트 스위칭: 5-10μs
```

## 왜 중요한가?
```
스레드 생성은 비싸다!
→ 매번 만들지 말고
→ 스레드 풀로 재사용!
```
```

---

## 슬라이드 21: 스레드 풀의 필요성
```markdown
# 스레드 풀 (Thread Pool)

## ❌ 매번 생성
```java
for (int i = 0; i < 1000; i++) {
    new Thread(() -> task()).start();
}
```
- 1000개 생성: ~1초
- 메모리: 1GB
- 컨텍스트 스위칭 폭증

## ✅ 스레드 풀 사용
```java
ExecutorService pool = Executors.newFixedThreadPool(20);
for (int i = 0; i < 1000; i++) {
    pool.submit(() -> task());
}
```
- 20개만 생성, 재사용
- 메모리: 20MB
- 효율적!

## 비유
```
매번 생성:   직원 1000명 면접 → 일 시킴 → 해고
스레드 풀:   직원 20명 고용 → 계속 일 시킴
```
```

---

## 슬라이드 22: 컨텍스트 스위칭
```markdown
# Context Switching (문맥 전환)

## 개념
```
CPU가 스레드를 전환할 때:
  Thread A 실행 중...
    ↓
  [저장] A의 상태 (Register, PC, Stack Pointer)
    ↓
  [복원] B의 상태
    ↓
  Thread B 실행 중...
```

## 비용
```
직접 비용:   5-10μs   (레지스터 저장/복원)
간접 비용:   50-100μs (CPU 캐시 미스)
```

## 문제
```
스레드가 너무 많으면?
→ 전환만 계속 함
→ 실제 일은 안 함
→ 오히려 느려짐!
```

## 실험 결과 (4코어 CPU, CPU-Bound 작업)
```
2 threads:   30ms  (1.6배)
4 threads:   20ms  (2.5배) ← 최적!
8 threads:   25ms  (2.0배) ← 느려짐
16 threads:  30ms  (1.6배) ← 더 느려짐
```

**교훈:** 스레드 많다고 빠른 게 아니다!
```

---

## 슬라이드 23: CPU-Bound vs I/O-Bound
```markdown
# 작업 유형별 전략

## CPU-Bound (계산 많음)
```
특징: CPU를 계속 사용
예시: 이미지 처리, 암호화, 정렬, 압축
```

**최적 스레드 수: 코어 수 + 1**
```
4코어 CPU → 5개 스레드

이유:
- CPU를 계속 쓰니까 많이 만들어봤자 경쟁
- 컨텍스트 스위칭만 증가
- 코어 수만큼이 최적
```

## I/O-Bound (대기 많음)
```
특징: I/O 대기 시간이 많음
예시: DB 쿼리, HTTP 요청, 파일 읽기
```

**최적 스레드 수: 코어 수 × (1 + 대기시간/CPU시간)**
```
4코어 CPU → 40~400개 스레드

이유:
- I/O 대기 중에는 CPU 안 씀
- 대기하는 동안 다른 스레드 실행 가능
- 많이 만들수록 효율적 (일정 수준까지)
```

## 계산 공식
```
최적 스레드 수 = 코어 수 × (1 + 대기시간 / CPU시간)

예: DB 쿼리 (대기 90%, CPU 10%)
  = 4 × (1 + 90/10)
  = 4 × 10
  = 40개
```

## 우리 서비스
```
댓글 좋아요: DB 쿼리 위주
→ I/O-Bound
→ 스레드 풀 크기를 충분히!
```
```

---

## 슬라이드 24: Java Memory Model (JMM)
```markdown
# Java Memory Model

## 구조
```
┌─────────────────────────────────┐
│         Main Memory             │
│  (Heap, Static Variables)       │
└─────────────────────────────────┘
       ↕              ↕
  ┌────────┐     ┌────────┐
  │ Cache1 │     │ Cache2 │
  │ (CPU1) │     │ (CPU2) │
  └────────┘     └────────┘
       ↕              ↕
  Thread 1        Thread 2
```

## 문제
```
Thread 1이 count = 1로 변경
  → Cache1에만 존재
  → Main Memory에 아직 안 씀
  → Thread 2는 count = 0으로 봄!
```

## 해결 방법

### synchronized
```java
synchronized(lock) {
    count++;  // Main Memory와 동기화
}
```

### volatile
```java
volatile int count;
// 항상 Main Memory에서 읽고 씀
```

### Atomic 클래스
```java
AtomicInteger count = new AtomicInteger(0);
count.incrementAndGet();  // 원자적 + 가시성
```
```

---

## 슬라이드 25: Virtual Thread (Java 21+)
```markdown
# Virtual Thread - 새로운 패러다임

## 기존 Platform Thread의 한계
```
Java Thread = OS Thread (1:1)
  → 생성 비용 높음 (~1ms, 1MB)
  → 수천 개는 부담
  → C10K Problem
```

## Virtual Thread의 혁신
```
Java Virtual Thread : OS Thread = N:1 또는 N:M
  → 생성 비용 낮음 (~1μs, 수백 bytes)
  → 수백만 개 가능!
```

## 구조
```
1,000,000 Virtual Threads
        ↓
     JVM Scheduler
        ↓
    10 Platform Threads (Carrier Thread)
        ↓
    10 OS Threads
```

## 코드
```java
// 기존 Platform Thread
Thread t = new Thread(() -> task());
t.start();

// Virtual Thread (Java 21+)
Thread t = Thread.ofVirtual().start(() -> task());

// ExecutorService
ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
executor.submit(() -> task());
```

## 언제 사용?
```
✅ I/O-Bound 작업 (DB, HTTP)
✅ 동시성 높은 서비스
❌ CPU-Bound 작업 (별 차이 없음)
```

## 주의사항
```
synchronized 블록에서 blocking 시 pinning 발생
→ ReentrantLock 사용 권장
```
```

---

## 슬라이드 26: 동시성 제어 도구 정리
```markdown
# 동시성 제어 방법 총정리

## Application Level

| 방법 | 원자성 | 가시성 | 사용처 |
|------|--------|--------|--------|
| synchronized | ✅ | ✅ | 간단한 동기화 |
| ReentrantLock | ✅ | ✅ | 복잡한 제어 |
| AtomicInteger | ✅ | ✅ | 단순 연산 (Lock-Free) |
| volatile | ❌ | ✅ | flag 변수 |

## Database Level

| 방법 | 특징 | 사용처 |
|------|------|--------|
| 비관적 락 | 미리 락 획득 | 충돌 많을 때 |
| 낙관적 락 | version 체크 | 충돌 적을 때 |

## 선택 기준
```
1. 작업 유형: Application vs Database
2. 충돌 빈도: 많음 vs 적음
3. 성능 요구사항: 속도 vs 안전성
4. 코드 복잡도: 단순 vs 복잡
```
```

---

## 슬라이드 27: Lock-Free 알고리즘
```markdown
# Lock-Free Programming

## CAS (Compare-And-Swap)
```java
// AtomicInteger 내부 동작
public final int incrementAndGet() {
    for (;;) {
        int current = get();
        int next = current + 1;
        if (compareAndSet(current, next))
            return next;
        // 실패하면 재시도
    }
}
```

## CAS 연산 (하드웨어 지원)
```
compareAndSet(expect, update):
  if (value == expect) {
      value = update;
      return true;
  }
  return false;

→ 원자적으로 실행! (CPU 명령어)
```

## 장점
```
✅ Lock 없음 → Deadlock 없음
✅ 대기 없음 → 빠름
✅ 확장성 좋음
```

## 단점
```
❌ 복잡함
❌ ABA Problem
❌ 단순 연산만 가능
```

## 언제 사용?
```
- 카운터, 플래그 같은 단순 변수
- 고성능 요구되는 경우
- Deadlock 회피 필요 시
```
```

---

## 슬라이드 28: Act 3 정리
```markdown
# 원리 파고들기 정리

## 깊이 이해한 것들

### 1. 락의 원리
- 원자성: 연산을 쪼개지지 않게
- 가시성: 변경사항을 모두가 보게

### 2. 스레드와 프로세스
- 프로세스: 독립적 (안전)
- 스레드: 공유 (동시성 문제)

### 3. Java Thread의 실체
- Java Thread = OS Thread (1:1)
- 생성 비용 높음 → 스레드 풀 필요

### 4. 성능 고려사항
- 컨텍스트 스위칭 비용
- CPU-Bound vs I/O-Bound
- 최적 스레드 수 계산

### 5. 최신 기술
- Virtual Thread (Java 21+)
- Lock-Free 알고리즘 (CAS)

## 핵심 깨달음
```
표면적 해결 → 깊은 이해
→ 다른 문제도 해결 가능!
→ 더 나은 선택 가능!
```
```

---

# Act 4: 이 지식으로 할 수 있는 것들 (8분, 8슬라이드)

## 슬라이드 29: Act 4 시작
```markdown
# 🚀 이제 할 수 있는 것들

## 지금까지 배운 것
- ✅ 동시성 문제의 본질 이해
- ✅ 다양한 해결 방법 알기
- ✅ 원리와 trade-off 이해

## 이제 할 수 있는 것
```
1. 상황에 맞는 동시성 제어 선택
2. 성능 최적화 패턴 적용
3. 문제 상황별 대응
4. 체크리스트로 예방
```
```

---

## 슬라이드 30: 상황별 동시성 제어 선택
```markdown
# Case Study: 어떤 방법을 선택할까?

## Case 1: 재고 차감
```java
상황: 주문 시 재고 1개 차감
충돌: 많음 (인기 상품은 동시 주문 많음)
중요도: 매우 높음 (재고 마이너스 되면 안 됨)
```
**선택: 비관적 락 (SELECT FOR UPDATE)**
```java
@Lock(LockModeType.PESSIMISTIC_WRITE)
Optional<Product> findById(Long id);
```

## Case 2: 조회수 증가
```java
상황: 게시글 조회 시 조회수 +1
충돌: 적음 (동시 조회 적음)
중요도: 낮음 (1~2개 차이는 괜찮음)
```
**선택: 낙관적 락 또는 AtomicInteger**
```java
@Version
private Long version;

// 또는
AtomicInteger viewCount = new AtomicInteger(0);
viewCount.incrementAndGet();
```

## Case 3: 카운터 (메모리)
```java
상황: API 호출 횟수 카운팅 (메모리)
충돌: 매우 많음
중요도: 높음
```
**선택: AtomicInteger (Lock-Free)**
```java
private final AtomicInteger apiCallCount = new AtomicInteger(0);
apiCallCount.incrementAndGet();
```

## Case 4: 설정 플래그
```java
상황: 서비스 활성화 on/off 플래그
충돌: 거의 없음 (관리자만 변경)
중요도: 가시성만 중요
```
**선택: volatile**
```java
private volatile boolean serviceEnabled = true;
```
```

---

## 슬라이드 31: 성능 최적화 패턴
```markdown
# 최적화 패턴 적용

## Pattern 1: 락 범위 최소화
```java
// ❌ 나쁜 예: 락 범위가 너무 넓음
synchronized void processOrder(Order order) {
    validateOrder(order);      // 100ms
    calculatePrice(order);     // 50ms
    updateInventory(order);    // 10ms  ← 이것만 동기화 필요
    sendEmail(order);          // 200ms
}

// ✅ 좋은 예: 필요한 부분만 락
void processOrder(Order order) {
    validateOrder(order);
    calculatePrice(order);

    synchronized(inventoryLock) {
        updateInventory(order);  // 10ms만 락
    }

    sendEmail(order);
}
```

## Pattern 2: 읽기-쓰기 락 분리
```java
// 읽기는 많고, 쓰기는 적은 경우
ReadWriteLock lock = new ReentrantReadWriteLock();

// 읽기: 동시 가능
public Data read() {
    lock.readLock().lock();
    try {
        return data;
    } finally {
        lock.readLock().unlock();
    }
}

// 쓰기: 배타적
public void write(Data newData) {
    lock.writeLock().lock();
    try {
        data = newData;
    } finally {
        lock.writeLock().unlock();
    }
}
```

## Pattern 3: 비동기 처리
```java
// ❌ 동기: 1000ms
public void processOrder(Order order) {
    checkStock(order);         // 300ms
    processPayment(order);     // 500ms
    reserveShipping(order);    // 200ms
}

// ✅ 비동기: 500ms (병렬 처리)
public CompletableFuture<Void> processOrder(Order order) {
    CompletableFuture<Void> stock =
        CompletableFuture.runAsync(() -> checkStock(order));
    CompletableFuture<Void> payment =
        CompletableFuture.runAsync(() -> processPayment(order));
    CompletableFuture<Void> shipping =
        CompletableFuture.runAsync(() -> reserveShipping(order));

    return CompletableFuture.allOf(stock, payment, shipping);
}
```
```

---

## 슬라이드 32: 배치 처리 패턴
```markdown
# Pattern 4: 배치 처리

## 문제
```
로그 저장: 요청마다 DB INSERT
→ 1000 requests = 1000 queries
→ DB 부하 높음
```

## 해결: 배치 처리
```java
@Component
public class LogBatchProcessor {
    private final BlockingQueue<LogEntry> queue =
        new LinkedBlockingQueue<>(10000);

    // 로그 추가 (빠름)
    public void addLog(LogEntry log) {
        queue.offer(log);  // Non-blocking
    }

    // 100ms마다 배치 처리
    @Scheduled(fixedRate = 100)
    public void processBatch() {
        List<LogEntry> batch = new ArrayList<>();
        queue.drainTo(batch, 1000);  // 최대 1000개

        if (!batch.isEmpty()) {
            logRepository.saveAll(batch);  // Bulk Insert
        }
    }
}
```

## 성능 향상
```
개별 처리: 10,000개 → 30초
배치 처리: 10,000개 → 2초 (15배 빨라짐!)
```

## 적용 사례
- 로그 저장
- 알림 발송
- 통계 집계
- 이벤트 처리
```

---

## 슬라이드 33: 문제 상황별 대응
```markdown
# 실제 문제 상황과 해결

## 상황 1: Deadlock 발생
```
증상: 특정 요청이 무한 대기
원인: Thread A → 락1, 락2 순서
      Thread B → 락2, 락1 순서 (반대!)
```

**해결:**
```java
// 항상 같은 순서로 락 획득
void transfer(Account from, Account to, int amount) {
    Account first = from.id < to.id ? from : to;
    Account second = from.id < to.id ? to : from;

    synchronized(first) {
        synchronized(second) {
            // 이체 로직
        }
    }
}
```

## 상황 2: 스레드 풀 포화
```
증상: 응답 시간 급증
원인: 스레드 수 부족, 작업 대기 큐 가득 참
```

**해결:**
```java
// 모니터링으로 감지
if (pool.getActiveCount() == pool.getMaximumPoolSize()) {
    alert("스레드 풀 포화!");
}

// 대응
1. 스레드 풀 크기 증가
2. 작업을 경량화
3. 비동기 처리 적용
4. Virtual Thread 도입 검토
```

## 상황 3: Memory Leak (ThreadLocal)
```
증상: 메모리 계속 증가
원인: ThreadLocal 정리 안 함
```

**해결:**
```java
// ❌ 위험
ThreadLocal<User> userContext = new ThreadLocal<>();

// ✅ 안전: 반드시 정리
try {
    userContext.set(user);
    // 사용...
} finally {
    userContext.remove();  // 필수!
}
```
```

---

## 슬라이드 34: 모니터링과 알림
```markdown
# 모니터링 전략

## 핵심 지표

### 스레드 풀 상태
```java
ThreadPoolExecutor pool = ...;

pool.getActiveCount()         // 활성 스레드
pool.getPoolSize()            // 현재 풀 크기
pool.getQueue().size()        // 대기 작업 수
pool.getCompletedTaskCount()  // 완료 작업 수
```

### 경고 조건
```
⚠️ activeCount == maxPoolSize  → 포화!
⚠️ queueSize > 80%             → 밀림!
⚠️ completedTaskCount 정체     → Deadlock?
```

## 실시간 알림
```java
@Scheduled(fixedRate = 5000)  // 5초마다
public void monitorThreadPool() {
    int active = pool.getActiveCount();
    int max = pool.getMaximumPoolSize();

    if (active == max) {
        alertService.send("스레드 풀 포화 경고!");
    }

    int queueSize = pool.getQueue().size();
    int queueCapacity = pool.getQueue().remainingCapacity();
    double usage = queueSize / (double)(queueSize + queueCapacity);

    if (usage > 0.8) {
        alertService.send("작업 큐 80% 초과!");
    }
}
```

## Grafana 대시보드
```
필수 패널:
- 활성 스레드 수 (시계열)
- 대기 작업 수 (시계열)
- 완료 작업 수 (카운터)
- 응답 시간 (히스토그램)
```
```

---

## 슬라이드 35: 체크리스트
```markdown
# 동시성 문제 예방 체크리스트

## 설계 단계
- [ ] 공유 자원을 명확히 파악했는가?
- [ ] 동시성 문제 가능성을 검토했는가?
- [ ] 적절한 동시성 제어 방법을 선택했는가?
- [ ] CPU-Bound vs I/O-Bound 파악했는가?
- [ ] 스레드 풀 크기를 계산했는가?

## 구현 단계
- [ ] 락 범위를 최소화했는가?
- [ ] Deadlock 가능성을 검토했는가?
- [ ] ThreadLocal 사용 시 정리 코드가 있는가?
- [ ] 예외 발생 시에도 락이 해제되는가?

## 테스트 단계
- [ ] 동시성 테스트를 작성했는가?
- [ ] 부하 테스트를 수행했는가?
- [ ] Deadlock 테스트를 했는가?
- [ ] 성능 프로파일링을 했는가?

## 운영 단계
- [ ] 스레드 풀 모니터링 중인가?
- [ ] 알림 설정이 되어 있는가?
- [ ] 로그는 충분한가?
- [ ] 문제 발생 시 대응 방안이 있는가?

## 문서화
- [ ] @ThreadSafe 또는 @NotThreadSafe 명시했는가?
- [ ] 락 순서를 문서화했는가?
- [ ] 스레드 풀 설정 근거를 문서화했는가?
```

---

## 슬라이드 36: 성능 측정과 최적화
```markdown
# 측정 없이는 최적화 없다

## JMH (Java Microbenchmark Harness)
```java
@Benchmark
@BenchmarkMode(Mode.Throughput)
public void testSynchronized() {
    synchronized(lock) {
        counter++;
    }
}

@Benchmark
public void testAtomic() {
    atomicCounter.incrementAndGet();
}
```

## 실제 측정 결과
```
Benchmark                    Mode  Cnt    Score   Error  Units
testSynchronized            thrpt   10  124.567 ± 2.345  ops/ms
testAtomic                  thrpt   10  345.123 ± 5.678  ops/ms
testLockFree (CAS)          thrpt   10  456.789 ± 6.789  ops/ms

→ Lock-Free가 3.7배 빠름!
```

## Thread Dump 분석
```bash
# Deadlock 감지
jstack <pid> | grep -A 10 "Found one Java-level deadlock"

# 대기 중인 스레드 확인
jstack <pid> | grep "WAITING" | wc -l
```

## VisualVM
```
- 실시간 스레드 상태 확인
- CPU 프로파일링
- 메모리 프로파일링
- Deadlock 감지
```
```

---

## 슬라이드 37: Act 4 정리
```markdown
# 이 지식으로 할 수 있는 것들 - 정리

## 배운 능력

### 1. 상황별 선택
- 재고 차감 → 비관적 락
- 조회수 증가 → 낙관적 락 or Atomic
- 카운터 → AtomicInteger
- 플래그 → volatile

### 2. 성능 최적화
- 락 범위 최소화
- 읽기-쓰기 락 분리
- 비동기 처리
- 배치 처리

### 3. 문제 해결
- Deadlock → 락 순서 일관성
- 스레드 풀 포화 → 크기 증가 or 비동기
- Memory Leak → ThreadLocal 정리

### 4. 예방과 모니터링
- 체크리스트로 예방
- 실시간 모니터링
- 알림 설정
- 성능 측정

## 핵심
```
원리를 이해했기 때문에
→ 다양한 상황에 대응 가능
→ 더 나은 선택 가능
→ 스스로 해결 가능
```
```

---

# 마무리 (2분, 3슬라이드)

## 슬라이드 38: 여정 정리
```markdown
# 우리가 걸어온 여정

## Act 1: 문제의 발견
```
평범한 코드 → 동시성 버그 발견
→ Race Condition
```

## Act 2: 해결의 과정
```
비관적 락 선택 → 적용 → 테스트
→ 1000개 정확히 처리 성공!
```

## Act 3: 원리를 파고들기 ⭐
```
왜 락이 동작하는가? (원자성, 가시성)
→ 스레드가 뭔데? (Java Thread = OS Thread)
→ 프로세스 vs 스레드 (메모리 구조)
→ 더 깊이 (컨텍스트 스위칭, 스레드 풀, Virtual Thread)
```

## Act 4: 이 지식으로 할 수 있는 것들
```
상황별 동시성 제어 선택
→ 성능 최적화 패턴
→ 문제 해결 전략
→ 예방 체크리스트
```

## 핵심 교훈
```
단순한 버그 픽스가 아니라
깊은 이해를 통한 진짜 성장!
```
```

---

## 슬라이드 39: 계속 공부하기
```markdown
# 🚀 더 깊이 공부하려면

## 추천 도서
```
1. "Java Concurrency in Practice" - Brian Goetz
   → 동시성의 바이블

2. "Operating System Concepts" - Silberschatz
   → OS 레벨 이해

3. "Effective Java" - Joshua Bloch
   → 78-84장 (동시성)
```

## 온라인 자료
```
- Oracle Java Tutorials: Concurrency
- Java Memory Model Spec (JSR-133)
- Virtual Thread (JEP 444)
- Project Loom
```

## 직접 해보기
```
1. 동시성 버그 직접 만들어보기
2. 다양한 락 방법으로 해결해보기
3. JMH로 성능 측정해보기
4. Virtual Thread 실험해보기
```

## 다음 학습 주제
```
- Lock-Free 자료구조 (ConcurrentHashMap 내부)
- 분산 시스템의 동시성 (Distributed Lock)
- Actor Model (Akka)
- Reactive Programming
```
```

---

## 슬라이드 40: Q&A
```markdown
# 질문 있으신가요? 💬

## 오늘 배운 것
```
✅ 동시성 문제의 본질
✅ 다양한 해결 방법
✅ 스레드와 프로세스의 원리
✅ 성능 최적화 전략
✅ 실제 적용 패턴
```

## 핵심 메시지
```
"문제를 만나면 깊이 파고들어라.
표면적 해결이 아닌 원리를 이해하면,
더 나은 개발자가 된다."
```

## 감사합니다! 🙏

```
발표 자료: [GitHub 링크]
연락처: [이메일]
```
```

---

# 📊 발표 가이드

## 시간 배분
```
Act 1: 7분  - 빠르게, 흥미 유발
Act 2: 7분  - 구체적으로, 코드 중심
Act 3: 18분 - 천천히, 깊이 있게 (핵심!)
Act 4: 8분  - 실용적으로, 적용 중심
마무리: 2분 - 메시지 전달
```

## 강조 포인트
```
Act 1: "이게 왜 문제일까?" - 궁금증 유발
Act 2: "이렇게 해결했다" - 성취감
Act 3: "원리를 파고들자" - 깊은 이해 (가장 중요!)
Act 4: "이제 이걸 할 수 있다" - 자신감
```

## 발표 톤
```
Act 1-2: 이야기하듯 편안하게
Act 3:   진지하게, 교육적으로 (하지만 지루하지 않게)
Act 4:   적극적으로, 실용적으로
```

---

**Good Luck! 🚀**
