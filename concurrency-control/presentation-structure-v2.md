# 동시성 문제 해결기: 버그부터 시스템 내부까지
## From Bug to System Internals

> **발표 컨셉**: 문제 → 해결 → 왜? → 시스템 내부까지! → 응용
> **발표 시간**: 40분
> **핵심**: OS/DB 개발자 관점으로 파고들기

---

## 🎬 발표 구조 Overview

```
Act 1: 문제의 발견 (7분)
    ↓
Act 2: 해결의 과정 (7분)
    ↓
Act 3: 원리를 파고들기 (20분) ← 핵심! 시스템 내부까지
    3-1~3-5: 기본 개념 (락, 트랜잭션, 스레드)
    3-6: OS 개발자 관점 - 커널의 스레드 관리 ⭐
    3-7: 스레드는 프로세스당 몇 개까지? ⭐
    3-8: 스레드 생명주기와 성능 진단 ⭐
    3-9: DB 개발자 관점 - 비관적 락 구현 ⭐
    ↓
Act 4: 이 지식으로 할 수 있는 것들 (6분)
    4-1: 병목 진단 능력
    4-2: 성능 최적화
    4-3: 아키텍처 설계
    ↓
마무리 (2분)
```

**핵심 스토리:**
```
버그 발견 → 비관적 락으로 해결 → 근데 왜 되는 거지?
→ 락/스레드 원리 → 더 깊이: OS는 어떻게 관리? DB는 어떻게 구현?
→ 이제 시스템 내부를 이해했으니 성능 최적화와 설계까지 가능!
```

---

# Act 1: 문제의 발견 (7분, 6슬라이드)

## 슬라이드 1: 타이틀
```markdown
동시성 문제 해결기
From Bug to System Internals

부제: 1000명이 동시에 좋아요를 누르면?

[이름]
[날짜]
```

---

## 슬라이드 2: 평범한 시작
```java
// 댓글 좋아요 기능
@Transactional
public void toggleLike(Long commentId, Long userId) {
    CommunityComment comment = repository.findById(commentId)
        .orElseThrow();

    comment.incrementLikeCount();  // count++
    repository.save(comment);
}

✅ 로컬 테스트: 정상
✅ 단위 테스트: 통과
✅ 배포 완료

"문제없어 보였다..."
```

---

## 슬라이드 3: 버그 발견
```markdown
# 🔴 부하 테스트 결과

시나리오: 1000명이 동시에 좋아요

예상: 1000
실제: 347, 523, 681... (매번 다름)

❓ "뭐가 문제지?"
```

---

## 슬라이드 4: Race Condition
```markdown
# 동시 실행 시나리오

Thread 1:  count = 0 읽기 → +1 계산 → count = 1 쓰기
Thread 2:  count = 0 읽기 → +1 계산 → count = 1 쓰기
                ↑ 동시에 0을 읽음!

결과: 1 (예상: 2)

**Race Condition 발생!**
```

---

## 슬라이드 5: count++ 의 비밀
```markdown
# count++ 는 3단계!

Java: count++

실제:
1. LOAD  count     // 읽기
2. ADD   1         // 계산
3. STORE count     // 쓰기

→ 중간에 끊기면 문제!
→ 원자성(Atomicity) 위반
```

---

## 슬라이드 6: 문제 정리
```markdown
# 발견한 것

1. 단일 사용자는 문제 없음
2. 동시 요청이 들어오면 데이터 깨짐
3. count++ 같은 단순 연산도 위험

## 원인
- Race Condition
- 원자성 문제

## 해결 과제
동시성 제어 (Concurrency Control)
```

---

# Act 2: 해결의 과정 (7분, 7슬라이드)

## 슬라이드 7: 비관적 락 선택
```markdown
# 동시성 제어 방법

Application Level: synchronized, Lock
Database Level: 비관적 락, 낙관적 락

## 우리의 선택: 비관적 락
- 충돌이 빈번할 것으로 예상
- 데이터 정합성 최우선
```

---

## 슬라이드 8: 해결 코드
```java
@Repository
public interface CommunityCommentRepository {
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT c FROM CommunityCommentEntity c WHERE c.commentId = :commentId")
    Optional<CommunityCommentEntity> findByIdWithPessimisticLock(@Param("commentId") Long commentId);
}
```

---

## 슬라이드 9: SQL 변환
```sql
SELECT *
FROM community_comment
WHERE comment_id = ?
FOR UPDATE;  -- ← 핵심!

FOR UPDATE의 의미:
- 배타적 락(Exclusive Lock)
- 트랜잭션 끝까지 다른 트랜잭션 대기
```

---

## 슬라이드 10: 동작 흐름
```markdown
Thread 1:
  락 획득 ✅ → count = 0 읽기 → count = 1 쓰기 → 커밋 & 락 해제

Thread 2:
  락 대기 ⏳ → 락 획득 ✅ → count = 1 읽기 (최신값!) → count = 2 쓰기

결과: 2 ✅ 정확!
```

---

## 슬라이드 11~13: (생략 - 테스트 코드 및 결과)

---

# Act 3: 원리를 파고들기 (20분, 16슬라이드)

## 슬라이드 14: 원리 파트 시작
```markdown
# 🤔 해결은 했는데...

궁금한 점:
1. 왜 락을 걸면 해결되는 거지?
2. 스레드가 뭐길래 이런 문제가?
3. OS는 스레드를 어떻게 관리하지?
4. DB는 락을 어떻게 구현하지?

## 이제부터
표면이 아닌 시스템 내부까지!
```

---

## 슬라이드 15: 락의 원리 - 원자성
```markdown
# 락이 해결하는 것 ① 원자성

문제: count++ 는 3단계
[락 없음] → 중간에 끊김

해결: 락으로 묶기
[락 획득]
  LOAD + ADD + STORE (하나로!)
[락 해제]

→ 원자성 보장!
```

---

## 슬라이드 16: 락의 원리 - 가시성
```markdown
# 락이 해결하는 것 ② 가시성

문제: CPU 캐시
Thread 1: count = 1 (캐시)
Thread 2: count = 0 (아직 안 보임)

해결:
트랜잭션 커밋 → DB(Main Memory)에 반영
다른 스레드가 최신값 읽기 가능

→ 가시성 보장!
```

---

## 슬라이드 17: 스레드란?
```markdown
# 스레드(Thread)

정의: CPU가 실행하는 가장 작은 단위

왜 필요?
한 번에 한 가지:  작업1 → 작업2 → 작업3
동시 처리:       작업1 ↘
                작업2 → 동시!
                작업3 ↗

예: 웹 서버 (Tomcat)
Thread 1~200: 사용자 200명 동시 처리
```

---

## 슬라이드 18: 프로세스 vs 스레드
```markdown
# 프로세스 vs 스레드

프로세스:
┌─────────────┐
│  Code       │
│  Heap       │
│  Stack      │
└─────────────┘
독립적!

스레드:
┌─────────────────┐
│ Code (공유)     │
│ Heap (공유) ⚠️  │
├─────────────────┤
│ Thread 1 Stack  │
│ Thread 2 Stack  │
└─────────────────┘

핵심: 공유 → 동시성 문제!
```

---

## 슬라이드 19: Java Thread = OS Thread
```markdown
# Java Thread = OS Thread

구조:
new Thread().start()
      ↓ JNI
    JVM
      ↓ System Call
  OS Kernel
      ↓
Native Thread 생성
      ↓
  1:1 매핑!

비용:
- 생성: ~1ms
- 메모리: ~1MB (Stack)
- 컨텍스트 스위칭: 5-10μs

→ 스레드는 비싸다!
```

---

## 슬라이드 20: ⭐ OS 개발자 관점 ① - 커널의 스레드 관리
```markdown
# 내가 OS 커널 개발자라면?

## 핵심 자료구조: TCB (Thread Control Block)

```c
struct TCB {
    int thread_id;              // 스레드 ID
    enum state;                 // Running, Ready, Blocked
    void* stack_pointer;        // Stack 위치
    void* instruction_pointer;  // 다음 실행 명령어 (PC)
    cpu_registers registers;    // 레지스터 백업
    int priority;               // 우선순위
};
```

## 스케줄러 로직

```
create_thread():
  1. TCB 메모리 할당
  2. Stack 공간 할당 (~1MB)
  3. state = Ready
  4. Ready Queue에 추가

context_switch(A → B):
  1. SAVE: Thread A의 레지스터 → TCB에 백업
  2. LOAD: Thread B의 TCB → 레지스터로 복원
  3. JUMP: Thread B의 명령어 주소로 점프
```

## '좋아요' 요청에서:
```
Thread A: DB 락 요청 → 이미 점유됨
커널: Thread A의 state를 Blocked로 변경
     → Wait Queue로 이동
     → CPU 할당 안 함 (CPU 절약!)

락 해제 시:
커널: Interrupt 받음
     → Thread A를 Ready Queue로 이동
     → 실행 기회 부여
```
```

---

## 슬라이드 21: ⭐ OS 개발자 관점 ② - 스레드 한계 (RAM)
```markdown
# 스레드는 프로세스당 몇 개까지?

## RAM의 물리적 한계

각 스레드 = Stack 공간 필요
- Java 기본: ~1MB per thread
- 스레드 1,000개 = 1GB 가상 메모리

계산:
```
최대 스레드 수 = (가용 메모리) / (스레드당 메모리)

32비트 OS: 프로세스당 2GB 제한
→ 최대 약 2,000개

64비트 OS: 이론적으론 무제한
→ 실제로는 물리 RAM 한계
```

커널 메모리도 소모:
- TCB 구조체
- 커널 스택

결론: 무한정 생성 불가!
→ ulimit으로 제한

## 왜 중요?
**용량 계획 (Capacity Planning)**
- 필요 메모리 = 최대 스레드 × 1MB
- 서버 사양 결정
- 컨테이너 리소스 제한 설정
```

---

## 슬라이드 22: ⭐ OS 개발자 관점 ③ - 스레드 한계 (CPU)
```markdown
# CPU 코어와 병렬성

## 진정한 병렬성

8코어 CPU = 특정 시점에 8개 스레드만 진짜 '동시' 실행

1,000개 스레드가 '동시'로 보이는 이유?
→ 컨텍스트 스위칭!

## 컨텍스트 스위칭 비용

직접 비용: 5-10μs (레지스터 저장/복원)
간접 비용: 50-100μs (CPU 캐시 미스)

## 스레싱 (Thrashing)

스레드 너무 많으면:
- 실제 일보다 전환에 시간 소모
- CPU 100%인데 느려짐

실험 결과 (4코어 CPU):
```
2 threads:   30ms  (1.6배)
4 threads:   20ms  (2.5배) ← 최적!
8 threads:   25ms  (2.0배) ← 느려짐
16 threads:  30ms  (1.6배) ← 더 느려짐
```

## 왜 중요?
**스레드 풀 크기 결정**
- newFixedThreadPool(32) ← 왜 32?
- 코어 수 × 작업 특성 고려
- 컨텍스트 스위칭 비용 최소화
```

---

## 슬라이드 23: ⭐ 스레드 생명주기와 성능 진단
```markdown
# 스레드 생명주기

```
New → Runnable → Running → Blocked/Waiting → Terminated
         ↑          ↓           ↓
         ←──────────←───────────←
```

## 각 상태의 의미

**Runnable**: CPU 받기를 기다림
**Running**: 실제 실행 중
**Blocked**: 외부 요인으로 강제 대기 ⚠️

## Blocked의 원인

1. **Lock 대기**
```java
synchronized(lock) {  // ← 여기서 Blocked!
    count++;
}
```

2. **I/O 대기**
```java
resultSet = statement.executeQuery();  // ← DB 응답 기다림
```

## 왜 중요? → 성능 진단!

**스레드 덤프 (Thread Dump)**
```
"http-nio-8080-exec-1" BLOCKED
  at CommentService.toggleLike(CommentService.java:42)
  - waiting to lock <0x000000076ab3d3e8>

"http-nio-8080-exec-2" BLOCKED
  at CommentService.toggleLike(CommentService.java:42)
  - waiting to lock <0x000000076ab3d3e8>

... (50개 스레드 모두 BLOCKED)
```

**진단:**
"아! CommentService의 락 경합(Lock Contention)이 병목이구나!"

**해결:**
- 트랜잭션 범위 최소화
- 락 방식 변경 (낙관적 락 고려)
- DB 쿼리 튜닝
```

---

## 슬라이드 24: ⭐ DB 개발자 관점 ① - 비관적 락 구현
```markdown
# 내가 DB 엔진(InnoDB) 개발자라면?

## 핵심 자료구조: Lock Manager

Hash Table:
```
Key: 잠글 대상 (table:row_id)
Value: {
  owner: 트랜잭션 ID
  type: Shared / Exclusive
  wait_queue: [대기 중인 트랜잭션들]
}
```

예:
```
'comment:5' → {
  owner: TX_100,
  type: Exclusive,
  wait_queue: [TX_101, TX_102, TX_103]
}
```
```

---

## 슬라이드 25: ⭐ DB 개발자 관점 ② - SELECT FOR UPDATE 흐름
```markdown
# SELECT ... FOR UPDATE 내부 동작

## 시나리오: TX_123이 comment_id=5에 락 요청

### CASE A: 락 없음
```
1. Lock Manager 조회: 'comment:5' → NULL
2. Hash Table에 추가:
   'comment:5' → {owner: TX_123, type: Exclusive, wait_queue: []}
3. 데이터 읽기 & 반환
```

### CASE B: 락 존재 (TX_100이 점유 중)
```
1. Lock Manager 조회: 'comment:5' → {owner: TX_100, ...}
2. Wait Queue에 추가:
   wait_queue: [TX_101, TX_102, TX_123]
3. TX_123의 DB 세션 → sleep() 호출
   → OS 스레드 → Blocked 상태
   → CPU 소모 없음!
```

### 락 해제 (TX_100 커밋)
```
1. TX_100: "내 모든 락 해제해줘"
2. Lock Manager:
   - 'comment:5'의 owner 제거
   - wait_queue 확인
   - 맨 앞 TX_101을 꺼내서 새 owner로 지정
   - TX_101 세션을 wakeup() 호출
   - OS 스레드 → Ready 상태로 전환
3. TX_101 작업 재개
```

## 왜 중요?
**Lock Wait 시간 최적화**
- 트랜잭션 짧게 유지 → 락 점유 시간 ↓
- 인덱스 최적화 → 락 대상 최소화
```

---

## 슬라이드 26: Act 3 정리
```markdown
# 원리 파고들기 정리

## 기본 이해
- 락의 원리 (원자성, 가시성)
- 스레드 vs 프로세스
- Java Thread = OS Thread

## 시스템 내부까지 ⭐
- **OS 커널**: TCB, 스케줄러, 컨텍스트 스위칭
- **스레드 한계**: RAM(Stack), CPU(병렬성)
- **생명주기**: Blocked 상태로 병목 진단
- **DB 엔진**: Lock Manager, 락 대기 큐

## 핵심 깨달음
```
표면적 이해 → 시스템 내부 이해
→ 성능 최적화 가능!
→ 아키텍처 설계 가능!
```
```

---

# Act 4: 이 지식으로 할 수 있는 것들 (6분, 6슬라이드)

## 슬라이드 27: Act 4 시작
```markdown
# 💡 이제 할 수 있는 것들

## 배운 지식
✅ 스레드 생명주기
✅ OS 커널의 스레드 관리
✅ RAM/CPU 한계
✅ DB Lock Manager

## 할 수 있는 것
1. 정확한 병목 진단
2. 근거 있는 성능 최적화
3. 확장 가능한 아키텍처 설계
```

---

## 슬라이드 28: 병목 진단 능력
```markdown
# 1. 정확한 병목 진단

## 증상
"서버가 느려요" (막연)

## 진단 과정

### 스레드 덤프 분석
```
jstack <pid> > thread_dump.txt
```

발견:
```
50개 스레드 → 모두 BLOCKED 상태
공통점: CommentService.toggleLike 메소드
원인: synchronized(lock) 대기
```

### DB 모니터링
```
SHOW ENGINE INNODB STATUS;
```

발견:
```
Lock Wait 이벤트 다수
comment_id=5에 Lock Contention
```

## 결론 (100% 확신!)
```
comment_id=5에 대한 비관적 락 경합이 병목!
```

## 왜 가능?
생명주기 지식 → Blocked 상태 이해
→ 스레드 덤프 해석 가능!
```

---

## 슬라이드 29: 성능 최적화 ① - 트랜잭션
```markdown
# 2. 성능 최적화 - 트랜잭션 범위

## 문제 코드
```java
@Transactional  // 락은 여기서 시작!
public void process() {
    Comment comment = repo.findByIdWithPessimisticLock(1L);
    comment.increaseLikeCount();

    externalApiCall();  // 3초 걸림! ← 락 3초 점유
} // 락 해제
```

## 개선 코드
```java
public void process() {
    updateLikeCount();  // DB 작업만 트랜잭션
    externalApiCall();  // 락과 무관
}

@Transactional
public void updateLikeCount() {
    Comment comment = repo.findByIdWithPessimisticLock(1L);
    comment.increaseLikeCount();
} // 락 즉시 해제! (0.01초)
```

## 결과
락 점유 시간: 3초 → 0.01초
처리량: 300배 향상!

## 왜 가능?
DB Lock Manager 지식
→ 락 점유 시간이 성능 직결!
```

---

## 슬라이드 30: 성능 최적화 ② - 스레드 풀
```markdown
# 2. 성능 최적화 - 스레드 풀 크기

## 계산 근거

### I/O-Bound 작업 (DB 쿼리)
```
최적 스레드 수 = 코어 수 × (1 + 대기시간/CPU시간)

예: DB 쿼리 (대기 90%, CPU 10%)
  = 4코어 × (1 + 90/10)
  = 4 × 10
  = 40개
```

### CPU-Bound 작업 (계산)
```
최적 스레드 수 = 코어 수 + 1

예: 4코어
  = 4 + 1
  = 5개
```

## 용량 계획
```
필요 메모리 = 40 스레드 × 1MB = 40MB (Stack)
서버 메모리 = Heap + Stack + OS = 2GB + 40MB + 500MB
```

## 왜 가능?
RAM/CPU 한계 지식
→ 정확한 계산 가능!
```

---

## 슬라이드 31: 아키텍처 설계
```markdown
# 3. 아키텍처 설계

## 문제 인식
```
여러 서버(Scale-out) → JPA @Lock 동작 안 함!
이유: DB 세션이 각 서버마다 다름
```

## 해결 ① 분산 락
```java
// Redis 분산 락
RLock lock = redisson.getLock("comment:5");
try {
    lock.lock(10, TimeUnit.SECONDS);
    // 여러 서버에 걸쳐 동시성 제어
} finally {
    lock.unlock();
}
```

## 해결 ② 락 회피 아키텍처
```
'좋아요' 요청 → Message Queue (Kafka)
                      ↓
                단일 Consumer (순차 처리)
                      ↓
                     DB

→ 락 없이도 정합성 보장!
```

## 왜 가능?
시스템 전체를 이해
→ 근본적인 설계 가능!
```

---

## 슬라이드 32: Act 4 정리
```markdown
# 할 수 있게 된 것들

## 1. 병목 진단
- 스레드 덤프 분석
- Blocked 상태로 원인 특정
- Lock Contention 식별

## 2. 성능 최적화
- 트랜잭션 범위 최소화
- 스레드 풀 크기 계산
- 메모리 용량 계획

## 3. 아키텍처 설계
- 분산 환경 고려
- 분산 락 도입
- 락 회피 아키텍처

## 핵심
```
시스템 내부를 이해했기 때문에
'감'이 아닌 '근거'로 최적화!
```
```

---

# 마무리 (2분, 2슬라이드)

## 슬라이드 33: 여정 요약
```markdown
# 우리가 걸어온 길

Act 1: 문제 발견
  → Race Condition

Act 2: 해결
  → 비관적 락

Act 3: 원리 파고들기 ⭐
  → 락/스레드 기본
  → OS 커널 내부 (TCB, 스케줄러)
  → 스레드 한계 (RAM, CPU)
  → 생명주기 (성능 진단)
  → DB 엔진 내부 (Lock Manager)

Act 4: 응용
  → 병목 진단
  → 성능 최적화
  → 아키텍처 설계

## 핵심 교훈
```
버그 수정을 넘어
시스템 내부까지 이해하는
진짜 성장!
```
```

---

## 슬라이드 34: Q&A
```markdown
# 질문 있으신가요? 💬

## 핵심 메시지
```
"문제를 만나면 표면에서 멈추지 말고
시스템 내부까지 파고들어라.

그 깊이가 곧
성능 최적화와 설계 역량이 된다."
```

## 감사합니다! 🙏
```

---

# 📊 발표 가이드

## 시간 배분
```
Act 1: 7분  - 빠르게, 흥미 유발
Act 2: 7분  - 구체적, 코드 중심
Act 3: 20분 - 천천히, 깊이! (핵심!)
  - 슬라이드 20~25가 하이라이트
Act 4: 6분  - 실용적, 적용 중심
마무리: 2분
```

## 강조 포인트
```
슬라이드 20: "내가 OS 개발자라면?" - 관점 전환
슬라이드 21-22: RAM/CPU 한계 - 구체적 숫자
슬라이드 23: Blocked 상태 - 성능 진단의 핵심
슬라이드 24-25: DB 내부 - Lock Manager 시각화
```

## 발표 톤
```
Act 1-2: 편안하게 이야기
Act 3 (20~25): 진지하게, 천천히 (하지만 흥미롭게!)
  "여러분이 만약 OS 개발자라면..."
  "커널은 이렇게 동작합니다"
Act 4: 자신감 있게
  "이제 우리는 이걸 할 수 있습니다!"
```

---

**Good Luck! 🚀**
