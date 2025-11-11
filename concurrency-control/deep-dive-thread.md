# 스레드 Deep Dive - 성능 향상을 위한 깊은 이해

> 스레드의 내부 동작 원리부터 실무 성능 최적화까지

---

## 목차

1. [스레드의 본질: OS 레벨 이해](#1-스레드의-본질-os-레벨-이해)
2. [JVM 스레드 구현의 비밀](#2-jvm-스레드-구현의-비밀)
3. [컨텍스트 스위칭의 비용](#3-컨텍스트-스위칭의-비용)
4. [스레드 풀: 왜, 언제, 어떻게](#4-스레드-풀-왜-언제-어떻게)
5. [동시성 vs 병렬성의 차이](#5-동시성-vs-병렬성의-차이)
6. [메모리 모델과 캐시 일관성](#6-메모리-모델과-캐시-일관성)
7. [Lock-Free 알고리즘](#7-lock-free-알고리즘)
8. [Virtual Thread (Java 21+)](#8-virtual-thread-java-21)
9. [성능 측정과 프로파일링](#9-성능-측정과-프로파일링)
10. [실무 성능 최적화 패턴](#10-실무-성능-최적화-패턴)

---

## 1. 스레드의 본질: OS 레벨 이해

### 1.1 스레드란 정확히 무엇인가?

**정의:**
- 스레드는 **CPU 스케줄링의 기본 단위**입니다.
- 프로세스 내에서 실행되는 **독립적인 실행 흐름**입니다.
- OS 커널이 관리하는 **실제 실행 컨텍스트**입니다.

**스레드의 구성 요소:**

```
┌─────────────────────────────────┐
│         Thread 구성             │
├─────────────────────────────────┤
│ 1. Thread ID (TID)              │  ← OS가 부여하는 고유 식별자
│ 2. Program Counter (PC)         │  ← 다음 실행할 명령어 위치
│ 3. Register Set                 │  ← CPU 레지스터 상태
│ 4. Stack Pointer (SP)           │  ← 스택의 현재 위치
│ 5. Stack Memory                 │  ← 독립적인 스택 공간
│ 6. Thread Local Storage (TLS)  │  ← 스레드 전용 저장소
│ 7. State (Running/Waiting/...)  │  ← 스레드 상태
└─────────────────────────────────┘
```

### 1.2 프로세스 vs 스레드 - 메모리 관점

```
┌───────────────────────────────────────────────────────┐
│                    Process A                          │
├───────────────────────────────────────────────────────┤
│  Code Segment        (공유)                            │
│  ┌─────────────────────────────────────┐              │
│  │  프로그램 명령어 (읽기 전용)           │              │
│  └─────────────────────────────────────┘              │
│                                                        │
│  Data Segment        (공유)                            │
│  ┌─────────────────────────────────────┐              │
│  │  전역 변수, static 변수               │              │
│  └─────────────────────────────────────┘              │
│                                                        │
│  Heap                (공유) ⚠️ 동시성 문제 발생 지점    │
│  ┌─────────────────────────────────────┐              │
│  │  동적 할당 메모리 (new, malloc)       │              │
│  │  • 객체                              │              │
│  │  • 배열                              │              │
│  │  • Collection                        │              │
│  └─────────────────────────────────────┘              │
│                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  Thread 1   │  │  Thread 2   │  │  Thread 3   │   │
│  │   Stack     │  │   Stack     │  │   Stack     │   │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤   │
│  │ 지역변수     │  │ 지역변수     │  │ 지역변수     │   │
│  │ 함수 호출    │  │ 함수 호출    │  │ 함수 호출    │   │
│  │ 매개변수     │  │ 매개변수     │  │ 매개변수     │   │
│  └─────────────┘  └─────────────┘  └─────────────┘   │
│        ↑                ↑                ↑             │
│     독립적           독립적            독립적           │
└───────────────────────────────────────────────────────┘
```

**핵심 포인트:**
- **Stack은 독립적** → 스레드 간 간섭 없음, 동시성 문제 없음
- **Heap은 공유** → 모든 스레드가 접근 가능, 동시성 문제의 근원

### 1.3 OS 스레드 스케줄링

**스케줄러의 역할:**

```
                    CPU Core 1        CPU Core 2
                        ↓                 ↓
                    ┌───────┐         ┌───────┐
Time 0-10ms         │ T1    │         │ T3    │
                    └───────┘         └───────┘
                        ↓                 ↓
Time 10-20ms        ┌───────┐         ┌───────┐
                    │ T2    │         │ T4    │
                    └───────┘         └───────┘
                        ↓                 ↓
Time 20-30ms        ┌───────┐         ┌───────┐
                    │ T1    │         │ T2    │
                    └───────┘         └───────┘

T1, T2, T3, T4 = 서로 다른 스레드
```

**스케줄링 알고리즘:**

1. **FCFS (First-Come, First-Served)**
   - 먼저 온 스레드를 먼저 처리
   - 간단하지만 비효율적 (Convoy Effect)

2. **Round Robin**
   - 시간 할당량(time quantum)을 정해서 순환
   - 대부분의 OS가 기본으로 사용

3. **Priority Scheduling**
   - 우선순위가 높은 스레드를 먼저 실행
   - Java의 `setPriority()`와 연결

4. **Multi-Level Queue**
   - 여러 큐를 두고 우선순위별로 관리
   - 실시간 OS에서 주로 사용

**중요한 사실:**
```java
// ⚠️ 개발자는 실행 순서를 제어할 수 없다!
Thread t1 = new Thread(() -> System.out.println("A"));
Thread t2 = new Thread(() -> System.out.println("B"));
t1.start();
t2.start();
// 출력이 "AB"일 수도, "BA"일 수도 있음
// 스케줄러가 결정!
```

### 1.4 스레드 상태 전이 (Thread State Transition)

```
                        start()
        NEW ──────────────────────→ RUNNABLE
                                        │
                                        │ 스케줄러가 선택
                                        ↓
                ┌───────────────── RUNNING ─────┐
                │                     │          │
                │ wait()              │          │ sleep()
                │ join()              │          │ I/O 대기
                ↓                     │          ↓
            WAITING              run() 완료   TIMED_WAITING
                │                     │          │
                │ notify()            │          │ 시간 만료
                │ notifyAll()         ↓          │
                └────────────→  TERMINATED  ←────┘

                        blocked on lock
        RUNNABLE ──────────────────→ BLOCKED
                        ←──────────────────
                        lock 획득
```

**각 상태의 의미:**

1. **NEW**: Thread 객체만 생성, 아직 OS 스레드 없음
2. **RUNNABLE**: OS 스레드 생성됨, 실행 대기 중
3. **RUNNING**: 실제로 CPU에서 실행 중
4. **BLOCKED**: 락 획득을 기다리는 중 (synchronized)
5. **WAITING**: 다른 스레드의 신호를 기다림 (wait, join)
6. **TIMED_WAITING**: 일정 시간 대기 (sleep, wait(timeout))
7. **TERMINATED**: 실행 완료

### 1.5 리눅스에서의 스레드 구현: NPTL

**Native POSIX Thread Library (NPTL):**

```bash
# 실제 리눅스에서 스레드 확인
$ ps -eLf | grep java
UID   PID  PPID   LWP  ... CMD
user  1234 1233  1234  ... java MyApp    # 메인 프로세스
user  1234 1233  1235  ... java MyApp    # 스레드 1
user  1234 1233  1236  ... java MyApp    # 스레드 2
user  1234 1233  1237  ... java MyApp    # 스레드 3

# LWP (Light Weight Process) = 리눅스에서의 스레드
```

**리눅스의 스레드 = 가벼운 프로세스:**
- `clone()` 시스템 콜로 생성
- 프로세스와 메모리 공유 (CLONE_VM 플래그)
- 독립적인 스택 할당

```c
// 리눅스 커널 레벨 (C 코드)
clone(fn, child_stack,
      CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND,
      arg);

// CLONE_VM: 메모리 공유 → 스레드
// CLONE_VM 없음: 메모리 독립 → 프로세스
```

---

## 2. JVM 스레드 구현의 비밀

### 2.1 Java Thread와 OS Thread의 매핑

**전통적인 1:1 매핑 (Java 20 이전):**

```
Java Application Layer
┌─────────────────────────────────┐
│  new Thread()                   │
│  new Thread()                   │
│  new Thread()                   │
└──────────┬──────────────────────┘
           │ JNI (Java Native Interface)
           ↓
JVM Layer
┌─────────────────────────────────┐
│  JavaThread 객체 생성           │
│  OS에 스레드 생성 요청           │
└──────────┬──────────────────────┘
           │ System Call
           ↓
OS Kernel Layer
┌─────────────────────────────────┐
│  pthread_create() (Linux)       │
│  CreateThread() (Windows)       │
│  → Native Thread 생성           │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│  1 Java Thread = 1 OS Thread    │
└─────────────────────────────────┘
```

**코드로 확인:**

```java
public class ThreadMappingDemo {
    public static void main(String[] args) {
        Thread t = new Thread(() -> {
            // 현재 스레드 정보
            long javaThreadId = Thread.currentThread().getId();

            // JVM 내부 정보 (Java 9+)
            ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
            ThreadInfo info = threadMXBean.getThreadInfo(javaThreadId);

            System.out.println("Java Thread ID: " + javaThreadId);
            System.out.println("Thread Name: " + info.getThreadName());
            System.out.println("Thread State: " + info.getThreadState());

            // OS 스레드 ID는 JNI나 /proc를 통해 확인 가능
            // Linux: /proc/[pid]/task/[tid]
        });

        t.start();
    }
}
```

### 2.2 스레드 생성 비용

**실제 측정:**

```java
public class ThreadCreationCost {
    public static void main(String[] args) throws InterruptedException {
        int threadCount = 10000;

        // 1. 스레드 직접 생성
        long start1 = System.nanoTime();
        for (int i = 0; i < threadCount; i++) {
            Thread t = new Thread(() -> {
                try { Thread.sleep(10); } catch (InterruptedException e) {}
            });
            t.start();
            t.join();
        }
        long end1 = System.nanoTime();
        System.out.println("직접 생성: " + (end1 - start1) / 1_000_000 + "ms");
        // 결과: 약 3000-5000ms

        // 2. 스레드 풀 사용
        ExecutorService executor = Executors.newFixedThreadPool(10);
        long start2 = System.nanoTime();
        for (int i = 0; i < threadCount; i++) {
            executor.submit(() -> {
                try { Thread.sleep(10); } catch (InterruptedException e) {}
            });
        }
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.MINUTES);
        long end2 = System.nanoTime();
        System.out.println("스레드 풀: " + (end2 - start2) / 1_000_000 + "ms");
        // 결과: 약 200-500ms (10배 이상 빠름!)
    }
}
```

**스레드 생성 시 발생하는 일:**

1. **JVM Heap에 Thread 객체 할당** (~500 bytes)
2. **OS 커널에 스레드 생성 요청** (시스템 콜)
3. **스택 메모리 할당** (기본 1MB, `-Xss` 옵션으로 조정)
4. **커널 자료구조 초기화** (task_struct 등)
5. **스케줄러에 등록**

**총 비용:**
- 시간: 약 0.2 ~ 1ms per thread
- 메모리: 1MB (스택) + 커널 메모리

### 2.3 스택 크기와 메모리 효율

```java
// 스택 크기 설정
// JVM 옵션: -Xss512k (기본은 1MB)

public class StackSizeDemo {
    private static int recursionDepth = 0;

    public static void deepRecursion() {
        recursionDepth++;
        deepRecursion(); // 재귀 호출
    }

    public static void main(String[] args) {
        try {
            deepRecursion();
        } catch (StackOverflowError e) {
            System.out.println("재귀 깊이: " + recursionDepth);
            // -Xss1m: 약 7000-10000
            // -Xss512k: 약 3500-5000
            // -Xss256k: 약 1700-2500
        }
    }
}
```

**스택 크기 선택 가이드:**

| 상황 | 권장 스택 크기 | 이유 |
|------|--------------|------|
| 일반 웹 애플리케이션 | 256KB - 512KB | 깊은 재귀 없음 |
| 수천 개 스레드 사용 | 128KB - 256KB | 메모리 절약 |
| 깊은 재귀/큰 지역변수 | 1MB - 2MB | StackOverflow 방지 |
| Virtual Thread | 자동 조정 | JVM이 최적화 |

**메모리 계산:**
```
1000개 스레드 × 1MB 스택 = 1GB 메모리 소비
1000개 스레드 × 256KB 스택 = 256MB 메모리 소비
→ 75% 메모리 절약!
```

### 2.4 GC와 스레드의 관계

**Stop-The-World (STW):**

```java
public class GCThreadDemo {
    public static void main(String[] args) {
        // 모든 애플리케이션 스레드가 멈춤!

        // GC 실행 중
        // ┌─────────────────────────────────┐
        // │  Application Threads (Stopped)  │
        // ├─────────────────────────────────┤
        // │  Thread 1: WAITING              │
        // │  Thread 2: WAITING              │
        // │  Thread 3: WAITING              │
        // └─────────────────────────────────┘
        //           ↓
        // ┌─────────────────────────────────┐
        // │  GC Threads (Running)           │
        // ├─────────────────────────────────┤
        // │  GC Thread 1: Mark              │
        // │  GC Thread 2: Sweep             │
        // │  GC Thread 3: Compact           │
        // └─────────────────────────────────┘

        // GC 완료 후
        // → 모든 애플리케이션 스레드 재개
    }
}
```

**GC 성능 최적화:**

```java
// JVM 옵션 예시
// -XX:+UseG1GC              : G1 GC 사용 (pause time 목표)
// -XX:MaxGCPauseMillis=200  : 최대 200ms pause 목표
// -XX:ParallelGCThreads=8   : GC 병렬 스레드 수
// -XX:ConcGCThreads=2       : 동시 GC 스레드 수
```

---

## 3. 컨텍스트 스위칭의 비용

### 3.1 컨텍스트 스위칭이란?

```
Time ──────────────────────────────────────────→

CPU: [Thread A] → [저장] → [복원] → [Thread B] → [저장] → [복원] → [Thread A]
         10ms      0.1ms     0.1ms      10ms       0.1ms     0.1ms      10ms
                   ↑─────── Context Switch ───────↑
```

**컨텍스트 스위칭 시 일어나는 일:**

1. **현재 스레드 상태 저장:**
   - Program Counter (PC)
   - CPU 레지스터 (범용 레지스터, 스택 포인터 등)
   - 스레드 상태 정보

2. **다음 스레드 상태 복원:**
   - 저장된 PC 복원
   - 저장된 레지스터 값 복원
   - MMU (Memory Management Unit) 설정

3. **캐시 무효화:**
   - CPU 캐시 (L1, L2, L3) 일부 무효화
   - TLB (Translation Lookaside Buffer) 플러시

**비용:**
- 직접 비용: 5-10 마이크로초 (시스템 콜 + 레지스터 저장/복원)
- 간접 비용: 50-100 마이크로초 (캐시 미스, TLB 미스)

### 3.2 컨텍스트 스위칭 측정

```java
public class ContextSwitchBenchmark {

    // 1. 단일 스레드 (컨텍스트 스위칭 없음)
    static long singleThreadBenchmark(int iterations) {
        long sum = 0;
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            sum += i;
        }
        long end = System.nanoTime();
        return end - start;
    }

    // 2. 멀티 스레드 (컨텍스트 스위칭 많음)
    static long multiThreadBenchmark(int iterations, int threadCount)
            throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        long start = System.nanoTime();

        CountDownLatch latch = new CountDownLatch(threadCount);
        for (int t = 0; t < threadCount; t++) {
            executor.submit(() -> {
                long sum = 0;
                for (int i = 0; i < iterations / threadCount; i++) {
                    sum += i;
                }
                latch.countDown();
            });
        }

        latch.await();
        long end = System.nanoTime();
        executor.shutdown();
        return end - start;
    }

    public static void main(String[] args) throws InterruptedException {
        int iterations = 10_000_000;

        long single = singleThreadBenchmark(iterations);
        System.out.println("Single Thread: " + single / 1_000_000 + "ms");

        long multi2 = multiThreadBenchmark(iterations, 2);
        System.out.println("2 Threads: " + multi2 / 1_000_000 + "ms");

        long multi4 = multiThreadBenchmark(iterations, 4);
        System.out.println("4 Threads: " + multi4 / 1_000_000 + "ms");

        long multi8 = multiThreadBenchmark(iterations, 8);
        System.out.println("8 Threads: " + multi8 / 1_000_000 + "ms");

        // 예상 결과 (4코어 CPU):
        // Single Thread: 50ms
        // 2 Threads: 30ms (1.6x 빨라짐, 이상적으론 2x)
        // 4 Threads: 20ms (2.5x 빨라짐, 이상적으론 4x)
        // 8 Threads: 25ms (2x 빨라짐, 오히려 느려짐!)
        //
        // → 스레드가 너무 많으면 컨텍스트 스위칭 오버헤드로 느려짐
    }
}
```

### 3.3 최적 스레드 수 계산

**공식 1: CPU-Bound 작업**
```
최적 스레드 수 = CPU 코어 수 + 1

이유: CPU를 최대한 활용하되, 컨텍스트 스위칭 최소화
```

**공식 2: I/O-Bound 작업**
```
최적 스레드 수 = CPU 코어 수 × (1 + Wait Time / Service Time)

예시:
- CPU Time: 10ms
- I/O Wait Time: 90ms
- CPU 코어: 4개

최적 스레드 수 = 4 × (1 + 90/10) = 4 × 10 = 40개
```

**실무 계산 코드:**

```java
public class OptimalThreadPoolSize {

    public static int calculateOptimalSize() {
        int cores = Runtime.getRuntime().availableProcessors();

        // CPU-Bound: 코어 수 + 1
        int cpuBound = cores + 1;

        // I/O-Bound: 측정 필요
        double cpuUtilization = 0.5; // 50% CPU 사용
        double waitTimeRatio = 0.9;  // 90% 대기 시간
        int ioBound = (int) (cores / cpuUtilization * (1 + waitTimeRatio));

        System.out.println("CPU Cores: " + cores);
        System.out.println("CPU-Bound 최적: " + cpuBound);
        System.out.println("I/O-Bound 최적: " + ioBound);

        return ioBound; // 보통 I/O-Bound가 더 많음
    }

    public static void main(String[] args) {
        calculateOptimalSize();
        // 4코어 CPU 예상 출력:
        // CPU Cores: 4
        // CPU-Bound 최적: 5
        // I/O-Bound 최적: 15-20
    }
}
```

---

## 4. 스레드 풀: 왜, 언제, 어떻게

### 4.1 스레드 풀의 필요성

**문제 상황:**

```java
// ❌ 나쁜 예: 요청마다 스레드 생성
@RestController
public class BadController {

    @PostMapping("/process")
    public Response process() {
        // 매 요청마다 새 스레드 생성!
        new Thread(() -> {
            // 무거운 작업
            heavyTask();
        }).start();

        return Response.ok();
    }
}

// 문제점:
// 1. 스레드 생성 비용 (1ms)
// 2. 메모리 낭비 (1MB per thread)
// 3. 동시 요청 1000개 = 1000개 스레드 = 1GB 메모리!
// 4. 컨텍스트 스위칭 폭증
// 5. CPU 오버헤드
```

**해결책: 스레드 풀**

```java
// ✅ 좋은 예: 스레드 풀 사용
@RestController
public class GoodController {

    private final ExecutorService threadPool =
        Executors.newFixedThreadPool(20); // 20개로 제한

    @PostMapping("/process")
    public Response process() {
        threadPool.submit(() -> {
            heavyTask();
        });
        return Response.ok();
    }
}

// 장점:
// 1. 스레드 재사용 → 생성 비용 절약
// 2. 메모리 제한 → 20MB만 사용
// 3. 컨텍스트 스위칭 감소
// 4. 리소스 제어 가능
```

### 4.2 ExecutorService 종류와 선택

```java
public class ExecutorServiceTypes {

    public static void main(String[] args) {

        // 1. newFixedThreadPool: 고정 크기 스레드 풀
        ExecutorService fixed = Executors.newFixedThreadPool(10);
        // 용도: 일정한 부하, CPU-Bound 작업
        // 특징: 스레드 수 고정, 큐는 무제한

        // 2. newCachedThreadPool: 필요시 생성, 유휴시 제거
        ExecutorService cached = Executors.newCachedThreadPool();
        // 용도: 짧고 많은 비동기 작업
        // 특징: 스레드 수 무제한, 60초 유휴시 제거
        // ⚠️ 위험: 스레드 폭증 가능

        // 3. newSingleThreadExecutor: 단일 스레드
        ExecutorService single = Executors.newSingleThreadExecutor();
        // 용도: 순차 처리 보장, 작업 큐잉
        // 특징: 한 번에 하나씩만 실행

        // 4. newScheduledThreadPool: 스케줄링
        ScheduledExecutorService scheduled =
            Executors.newScheduledThreadPool(5);
        // 용도: 주기적 작업, 지연 실행
        scheduled.scheduleAtFixedRate(() -> {
            System.out.println("5초마다 실행");
        }, 0, 5, TimeUnit.SECONDS);

        // 5. newWorkStealingPool: Work Stealing (Java 8+)
        ExecutorService workStealing =
            Executors.newWorkStealingPool();
        // 용도: CPU-Bound, 재귀 작업 (Fork/Join)
        // 특징: 각 스레드가 큐를 가지고, 유휴 스레드가 다른 큐에서 작업 훔쳐옴

        // 6. ThreadPoolExecutor: 커스텀 설정 (⭐ 실무 권장)
        ThreadPoolExecutor custom = new ThreadPoolExecutor(
            10,                      // corePoolSize: 기본 스레드 수
            20,                      // maximumPoolSize: 최대 스레드 수
            60L,                     // keepAliveTime: 유휴 스레드 생존 시간
            TimeUnit.SECONDS,        // 시간 단위
            new LinkedBlockingQueue<>(100),  // 작업 큐 (크기 제한!)
            new ThreadPoolExecutor.CallerRunsPolicy()  // 거부 정책
        );

        // 종료
        fixed.shutdown();
        cached.shutdown();
        single.shutdown();
        scheduled.shutdown();
        workStealing.shutdown();
        custom.shutdown();
    }
}
```

### 4.3 ThreadPoolExecutor 상세 설정

**핵심 파라미터 이해:**

```java
public class ThreadPoolConfiguration {

    public static ThreadPoolExecutor createOptimizedPool() {
        int cores = Runtime.getRuntime().availableProcessors();

        return new ThreadPoolExecutor(
            // 1. corePoolSize (핵심 스레드 수)
            cores * 2,
            // → 항상 살아있는 스레드
            // → I/O-Bound: cores * 2
            // → CPU-Bound: cores + 1

            // 2. maximumPoolSize (최대 스레드 수)
            cores * 4,
            // → 부하가 높을 때 추가 생성할 최대 스레드
            // → 큐가 가득 차면 이 수까지 생성

            // 3. keepAliveTime (유휴 스레드 생존 시간)
            60L,
            TimeUnit.SECONDS,
            // → core를 초과하는 스레드의 최대 유휴 시간
            // → 60초 동안 작업 없으면 스레드 제거

            // 4. workQueue (작업 큐)
            new ArrayBlockingQueue<>(100),
            // 옵션:
            // - LinkedBlockingQueue: 무제한 큐 (⚠️ 메모리 위험)
            // - ArrayBlockingQueue: 고정 크기 (✅ 권장)
            // - SynchronousQueue: 큐 없음, 직접 전달
            // - PriorityBlockingQueue: 우선순위 큐

            // 5. threadFactory (스레드 팩토리)
            new ThreadFactory() {
                private AtomicInteger count = new AtomicInteger(1);

                @Override
                public Thread newThread(Runnable r) {
                    Thread thread = new Thread(r);
                    thread.setName("MyPool-" + count.getAndIncrement());
                    thread.setDaemon(false); // 데몬 스레드 여부
                    thread.setPriority(Thread.NORM_PRIORITY);
                    return thread;
                }
            },

            // 6. rejectedExecutionHandler (거부 정책)
            new ThreadPoolExecutor.CallerRunsPolicy()
            // 옵션:
            // - AbortPolicy: 예외 던짐 (기본값)
            // - CallerRunsPolicy: 호출 스레드에서 실행 (✅ 권장)
            // - DiscardPolicy: 조용히 버림
            // - DiscardOldestPolicy: 가장 오래된 작업 버리고 재시도
        );
    }

    public static void main(String[] args) {
        ThreadPoolExecutor pool = createOptimizedPool();

        // 모니터링
        System.out.println("Active Threads: " + pool.getActiveCount());
        System.out.println("Pool Size: " + pool.getPoolSize());
        System.out.println("Queue Size: " + pool.getQueue().size());
        System.out.println("Completed Tasks: " + pool.getCompletedTaskCount());

        pool.shutdown();
    }
}
```

### 4.4 실무 스레드 풀 패턴

**패턴 1: 계층형 스레드 풀**

```java
@Configuration
public class ThreadPoolConfig {

    // API 요청 처리용 (빠른 응답 필요)
    @Bean(name = "apiThreadPool")
    public ThreadPoolTaskExecutor apiThreadPool() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(20);
        executor.setMaxPoolSize(40);
        executor.setQueueCapacity(100);
        executor.setThreadNamePrefix("API-");
        executor.setRejectedExecutionHandler(
            new ThreadPoolExecutor.CallerRunsPolicy());
        executor.initialize();
        return executor;
    }

    // 백그라운드 작업용 (느려도 됨)
    @Bean(name = "backgroundThreadPool")
    public ThreadPoolTaskExecutor backgroundThreadPool() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(5);
        executor.setMaxPoolSize(10);
        executor.setQueueCapacity(1000);
        executor.setThreadNamePrefix("BG-");
        executor.setRejectedExecutionHandler(
            new ThreadPoolExecutor.CallerRunsPolicy());
        executor.initialize();
        return executor;
    }

    // 스케줄링 작업용
    @Bean(name = "scheduledThreadPool")
    public TaskScheduler scheduledThreadPool() {
        ThreadPoolTaskScheduler scheduler = new ThreadPoolTaskScheduler();
        scheduler.setPoolSize(5);
        scheduler.setThreadNamePrefix("Scheduled-");
        scheduler.initialize();
        return scheduler;
    }
}

@Service
public class TaskService {

    @Autowired
    @Qualifier("apiThreadPool")
    private ThreadPoolTaskExecutor apiPool;

    @Autowired
    @Qualifier("backgroundThreadPool")
    private ThreadPoolTaskExecutor bgPool;

    public void handleApiRequest() {
        apiPool.submit(() -> {
            // 빠른 처리
        });
    }

    public void handleBackgroundTask() {
        bgPool.submit(() -> {
            // 느린 처리
        });
    }
}
```

**패턴 2: 동적 스레드 풀 조정**

```java
public class AdaptiveThreadPool {

    private ThreadPoolExecutor pool;

    public AdaptiveThreadPool() {
        int cores = Runtime.getRuntime().availableProcessors();
        this.pool = new ThreadPoolExecutor(
            cores, cores * 4, 60L, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(100)
        );

        // 5초마다 모니터링하고 조정
        ScheduledExecutorService monitor =
            Executors.newSingleThreadScheduledExecutor();
        monitor.scheduleAtFixedRate(this::adjustPoolSize, 0, 5, TimeUnit.SECONDS);
    }

    private void adjustPoolSize() {
        int active = pool.getActiveCount();
        int poolSize = pool.getPoolSize();
        int queueSize = pool.getQueue().size();

        // 큐가 80% 이상 차면 스레드 증가
        if (queueSize > 80 && poolSize < pool.getMaximumPoolSize()) {
            pool.setCorePoolSize(poolSize + 2);
            System.out.println("스레드 풀 증가: " + poolSize + " → " + (poolSize + 2));
        }

        // 활성 스레드가 20% 미만이면 감소
        if (active < poolSize * 0.2 && poolSize > pool.getCorePoolSize()) {
            pool.setCorePoolSize(poolSize - 2);
            System.out.println("스레드 풀 감소: " + poolSize + " → " + (poolSize - 2));
        }
    }
}
```

---

## 5. 동시성 vs 병렬성의 차이

### 5.1 개념 구분

**동시성 (Concurrency):**
```
단일 코어에서 여러 작업을 번갈아가며 실행

Time ────────────────────────────────→
     [A][B][A][C][B][A][C][B]...

한 번에 하나씩 실행하지만, 빠르게 전환해서 동시에 실행되는 것처럼 보임
```

**병렬성 (Parallelism):**
```
여러 코어에서 여러 작업을 실제로 동시에 실행

Core 1: [AAAAAAAA]
Core 2: [BBBBBBBB]
Core 3: [CCCCCCCC]
Time ──→

실제로 동시에 실행됨
```

**코드 예시:**

```java
public class ConcurrencyVsParallelism {

    // 동시성: 싱글 코어에서도 동작
    public static void concurrencyExample() {
        // 여러 스레드가 번갈아가며 실행
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("Task A-" + i);
                try { Thread.sleep(100); } catch (InterruptedException e) {}
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("Task B-" + i);
                try { Thread.sleep(100); } catch (InterruptedException e) {}
            }
        });

        t1.start();
        t2.start();

        // 출력: A-0, B-0, A-1, B-1, ... (섞임)
        // → 동시성: 빠르게 전환
    }

    // 병렬성: 멀티 코어에서 진정한 병렬 실행
    public static void parallelismExample() {
        List<Integer> numbers = IntStream.rangeClosed(1, 1000000)
            .boxed()
            .collect(Collectors.toList());

        // 순차 처리
        long start1 = System.currentTimeMillis();
        long sum1 = numbers.stream()
            .mapToLong(i -> i * i)
            .sum();
        long end1 = System.currentTimeMillis();
        System.out.println("순차: " + (end1 - start1) + "ms");

        // 병렬 처리 (여러 코어 사용)
        long start2 = System.currentTimeMillis();
        long sum2 = numbers.parallelStream()
            .mapToLong(i -> i * i)
            .sum();
        long end2 = System.currentTimeMillis();
        System.out.println("병렬: " + (end2 - start2) + "ms");

        // 4코어 CPU에서 결과:
        // 순차: 100ms
        // 병렬: 30ms (약 3.3배 빨라짐)
    }
}
```

### 5.2 CPU-Bound vs I/O-Bound

**CPU-Bound 작업:**
```java
// CPU를 많이 사용하는 작업
public class CPUBoundTask {

    public static long fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    public static void benchmark() {
        int cores = Runtime.getRuntime().availableProcessors();

        // 1. 싱글 스레드
        long start1 = System.currentTimeMillis();
        for (int i = 0; i < 40; i++) {
            fibonacci(30);
        }
        long end1 = System.currentTimeMillis();
        System.out.println("싱글: " + (end1 - start1) + "ms");

        // 2. 코어 수만큼 스레드
        ExecutorService pool = Executors.newFixedThreadPool(cores);
        long start2 = System.currentTimeMillis();
        List<Future<?>> futures = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            futures.add(pool.submit(() -> fibonacci(30)));
        }
        futures.forEach(f -> {
            try { f.get(); } catch (Exception e) {}
        });
        long end2 = System.currentTimeMillis();
        System.out.println("병렬 (" + cores + "코어): " + (end2 - start2) + "ms");

        pool.shutdown();

        // 4코어 결과:
        // 싱글: 2000ms
        // 병렬 (4코어): 550ms (약 3.6배 빨라짐)
        //
        // ⭐ CPU-Bound는 코어 수만큼만 병렬화 효과!
    }
}
```

**I/O-Bound 작업:**
```java
// I/O 대기가 많은 작업
public class IOBoundTask {

    public static String fetchUrl(String url) throws Exception {
        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .build();
        HttpResponse<String> response = client.send(request,
            HttpResponse.BodyHandlers.ofString());
        return response.body();
    }

    public static void benchmark() throws Exception {
        List<String> urls = List.of(
            "https://api.github.com/users/github",
            "https://api.github.com/users/google",
            "https://api.github.com/users/microsoft"
            // ... 100개 URL
        );

        // 1. 순차 처리
        long start1 = System.currentTimeMillis();
        for (String url : urls) {
            fetchUrl(url);
        }
        long end1 = System.currentTimeMillis();
        System.out.println("순차: " + (end1 - start1) + "ms");

        // 2. 병렬 처리 (50개 스레드)
        ExecutorService pool = Executors.newFixedThreadPool(50);
        long start2 = System.currentTimeMillis();
        List<Future<String>> futures = new ArrayList<>();
        for (String url : urls) {
            futures.add(pool.submit(() -> fetchUrl(url)));
        }
        for (Future<String> f : futures) {
            f.get();
        }
        long end2 = System.currentTimeMillis();
        System.out.println("병렬 (50스레드): " + (end2 - start2) + "ms");

        pool.shutdown();

        // 결과:
        // 순차: 10000ms (각 100ms × 100개)
        // 병렬 (50스레드): 250ms (40배 빨라짐!)
        //
        // ⭐ I/O-Bound는 코어 수보다 훨씬 많은 스레드로 효과!
    }
}
```

**선택 가이드:**

| 작업 유형 | 최적 스레드 수 | 이유 |
|----------|--------------|------|
| CPU-Bound | 코어 수 + 1 | 컨텍스트 스위칭 최소화 |
| I/O-Bound | 코어 수 × 10 ~ 100 | 대기 시간 활용 |
| 혼합 | 측정 필요 | 프로파일링으로 결정 |

---

## 6. 메모리 모델과 캐시 일관성

### 6.1 Java Memory Model (JMM)

**메모리 구조:**

```
┌─────────────────────────────────────────────────────┐
│                   Main Memory (RAM)                 │
│  ┌──────────────────────────────────────────────┐   │
│  │  Heap: 공유 객체, 배열, static 변수          │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────┬───────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
┌─────────────┐     ┌─────────────┐
│  CPU Core 1 │     │  CPU Core 2 │
├─────────────┤     ├─────────────┤
│  L1 Cache   │     │  L1 Cache   │  ← 32-64KB, 1ns
│  (32KB)     │     │  (32KB)     │
├─────────────┤     ├─────────────┤
│  L2 Cache   │     │  L2 Cache   │  ← 256KB, 3-10ns
│  (256KB)    │     │  (256KB)    │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 ↓
         ┌───────────────┐
         │   L3 Cache    │  ← 8-32MB, 10-20ns
         │   (8MB 공유)   │
         └───────────────┘
                 ↓
         Main Memory (100ns)

┌─────────────┐     ┌─────────────┐
│  Thread 1   │     │  Thread 2   │
│  Stack      │     │  Stack      │  ← 지역변수, 독립적
└─────────────┘     └─────────────┘
```

### 6.2 가시성 문제 (Visibility Problem)

```java
public class VisibilityProblem {

    // ❌ 문제: 가시성 보장 안 됨
    private static boolean flag = false;

    public static void problemExample() {
        Thread writer = new Thread(() -> {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {}
            flag = true;  // Main Memory에 언제 쓰일지 모름
            System.out.println("Writer: flag = true");
        });

        Thread reader = new Thread(() -> {
            while (!flag) {  // CPU 캐시에서 계속 false 읽을 수 있음
                // 무한 루프 가능!
            }
            System.out.println("Reader: flag is true!");
        });

        reader.start();
        writer.start();

        // 예상: Writer가 flag=true 설정 → Reader 종료
        // 실제: Reader가 무한 루프에 빠질 수 있음!
    }

    // ✅ 해결 1: volatile
    private static volatile boolean flagVolatile = false;

    public static void volatileSolution() {
        Thread writer = new Thread(() -> {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {}
            flagVolatile = true;  // Main Memory에 즉시 반영
            System.out.println("Writer: flag = true");
        });

        Thread reader = new Thread(() -> {
            while (!flagVolatile) {  // Main Memory에서 직접 읽음
                // 정상 종료!
            }
            System.out.println("Reader: flag is true!");
        });

        reader.start();
        writer.start();
    }

    // ✅ 해결 2: synchronized
    private static boolean flagSync = false;
    private static final Object lock = new Object();

    public static void synchronizedSolution() {
        Thread writer = new Thread(() -> {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {}
            synchronized (lock) {
                flagSync = true;  // synchronized 진입 시 캐시 동기화
            }
            System.out.println("Writer: flag = true");
        });

        Thread reader = new Thread(() -> {
            boolean localFlag = false;
            while (!localFlag) {
                synchronized (lock) {
                    localFlag = flagSync;  // synchronized 진입 시 최신값 읽음
                }
            }
            System.out.println("Reader: flag is true!");
        });

        reader.start();
        writer.start();
    }
}
```

### 6.3 Happens-Before 관계

**개념:**
- A happens-before B → A의 결과가 B에게 보임
- JMM의 핵심 개념

**Happens-Before 규칙:**

```java
public class HappensBeforeExample {

    private int x = 0;
    private volatile boolean ready = false;

    // Rule 1: 프로그램 순서 규칙
    // 같은 스레드 내에서 앞의 명령어는 뒤의 명령어보다 먼저 발생
    public void rule1() {
        int a = 1;  // happens-before
        int b = 2;  // b는 a=1을 볼 수 있음
    }

    // Rule 2: Monitor Lock 규칙
    // unlock은 다음 lock보다 먼저 발생
    private final Object lock = new Object();

    public void rule2Writer() {
        synchronized (lock) {
            x = 42;
        }  // unlock happens-before 다음 lock
    }

    public void rule2Reader() {
        synchronized (lock) {  // lock
            System.out.println(x);  // 42를 볼 수 있음
        }
    }

    // Rule 3: volatile 변수 규칙
    // volatile 쓰기는 다음 volatile 읽기보다 먼저 발생
    public void rule3Writer() {
        x = 42;         // (1)
        ready = true;   // (2) volatile write
    }

    public void rule3Reader() {
        if (ready) {    // (3) volatile read
            System.out.println(x);  // (4) 42를 볼 수 있음
        }
        // (2) happens-before (3)
        // (1) happens-before (2) (프로그램 순서)
        // → (1) happens-before (4)
    }

    // Rule 4: Thread.start() 규칙
    // start() 호출은 시작된 스레드의 모든 명령어보다 먼저 발생
    public void rule4() {
        x = 42;  // (1)
        Thread t = new Thread(() -> {
            System.out.println(x);  // (2) 42를 볼 수 있음
        });
        t.start();  // (1) happens-before (2)
    }

    // Rule 5: Thread.join() 규칙
    // 스레드의 모든 명령어는 join() 반환보다 먼저 발생
    public void rule5() throws InterruptedException {
        Thread t = new Thread(() -> {
            x = 42;  // (1)
        });
        t.start();
        t.join();  // (1) happens-before join() 반환
        System.out.println(x);  // (2) 42를 볼 수 있음
    }
}
```

### 6.4 False Sharing 문제

**개념:**
- CPU 캐시 라인은 보통 64 bytes
- 서로 다른 변수가 같은 캐시 라인에 있으면 간섭 발생

```java
public class FalseSharingProblem {

    // ❌ 문제: False Sharing 발생
    static class BadCounter {
        public long count1 = 0;  // 8 bytes
        public long count2 = 0;  // 8 bytes (같은 캐시 라인!)
    }

    public static void badExample() throws InterruptedException {
        BadCounter counter = new BadCounter();

        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 100_000_000; i++) {
                counter.count1++;  // 캐시 라인 무효화
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 100_000_000; i++) {
                counter.count2++;  // 캐시 라인 무효화
            }
        });

        long start = System.currentTimeMillis();
        t1.start();
        t2.start();
        t1.join();
        t2.join();
        long end = System.currentTimeMillis();

        System.out.println("Bad: " + (end - start) + "ms");
        // 결과: 약 3000ms
    }

    // ✅ 해결: 패딩으로 캐시 라인 분리
    static class GoodCounter {
        public long count1 = 0;
        public long p1, p2, p3, p4, p5, p6, p7;  // 패딩 (56 bytes)
        public long count2 = 0;  // 다른 캐시 라인!
        public long p8, p9, p10, p11, p12, p13, p14;  // 패딩
    }

    public static void goodExample() throws InterruptedException {
        GoodCounter counter = new GoodCounter();

        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 100_000_000; i++) {
                counter.count1++;  // 독립적인 캐시 라인
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 100_000_000; i++) {
                counter.count2++;  // 독립적인 캐시 라인
            }
        });

        long start = System.currentTimeMillis();
        t1.start();
        t2.start();
        t1.join();
        t2.join();
        long end = System.currentTimeMillis();

        System.out.println("Good: " + (end - start) + "ms");
        // 결과: 약 800ms (약 4배 빨라짐!)
    }

    // ✅ Java 8+: @Contended 어노테이션
    // JVM 옵션 필요: -XX:-RestrictContended
    static class BestCounter {
        @sun.misc.Contended
        public long count1 = 0;

        @sun.misc.Contended
        public long count2 = 0;
    }
}
```

---

## 7. Lock-Free 알고리즘

### 7.1 CAS (Compare-And-Swap)

**개념:**
```
CAS(memory, expected, new):
    if memory == expected:
        memory = new
        return true
    else:
        return false

원자적으로 실행! (CPU 명령어 레벨)
```

**Java 구현:**

```java
public class CASExample {

    private AtomicInteger count = new AtomicInteger(0);

    // AtomicInteger 내부 동작 원리
    public int incrementAndGet() {
        int current;
        int next;
        do {
            current = count.get();      // 현재 값 읽기
            next = current + 1;         // 새 값 계산
        } while (!count.compareAndSet(current, next));  // CAS
        // current가 여전히 같으면 next로 변경, 성공
        // 다른 스레드가 수정했으면 실패, 재시도

        return next;
    }

    // 실제 사용
    public static void benchmark() throws InterruptedException {
        AtomicInteger atomic = new AtomicInteger(0);
        int threadCount = 10;
        int iterations = 1_000_000;

        Thread[] threads = new Thread[threadCount];
        long start = System.currentTimeMillis();

        for (int i = 0; i < threadCount; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < iterations; j++) {
                    atomic.incrementAndGet();
                }
            });
            threads[i].start();
        }

        for (Thread t : threads) {
            t.join();
        }

        long end = System.currentTimeMillis();
        System.out.println("AtomicInteger: " + (end - start) + "ms");
        System.out.println("Final count: " + atomic.get());

        // 결과:
        // AtomicInteger: 500ms
        // Final count: 10000000 (정확!)
        //
        // synchronized와 비교:
        // - AtomicInteger: 500ms (Lock-Free)
        // - synchronized: 800ms (Lock 오버헤드)
    }
}
```

### 7.2 ABA 문제

```java
public class ABAProblem {

    static class Node {
        int value;
        Node next;

        Node(int value) {
            this.value = value;
        }
    }

    // ❌ ABA 문제 예시
    static class ProblemStack {
        private AtomicReference<Node> head = new AtomicReference<>();

        public void push(int value) {
            Node newNode = new Node(value);
            Node oldHead;
            do {
                oldHead = head.get();
                newNode.next = oldHead;
            } while (!head.compareAndSet(oldHead, newNode));
        }

        public Integer pop() {
            Node oldHead;
            Node newHead;
            do {
                oldHead = head.get();
                if (oldHead == null) return null;
                newHead = oldHead.next;
                // ⚠️ 여기서 문제 발생 가능!
                // Thread 1이 oldHead를 읽음 (A)
                // Thread 2가 pop() → head가 B로 변경
                // Thread 3가 push(A) → head가 다시 A로 변경
                // Thread 1의 CAS 성공! (하지만 잘못된 상태)
            } while (!head.compareAndSet(oldHead, newHead));
            return oldHead.value;
        }
    }

    // ✅ 해결: AtomicStampedReference (버전 번호 추가)
    static class SolutionStack {
        private AtomicStampedReference<Node> head =
            new AtomicStampedReference<>(null, 0);

        public void push(int value) {
            Node newNode = new Node(value);
            int[] stampHolder = new int[1];
            Node oldHead;
            do {
                oldHead = head.get(stampHolder);
                int stamp = stampHolder[0];
                newNode.next = oldHead;
            } while (!head.compareAndSet(
                oldHead, newNode, stamp, stamp + 1));  // 버전도 체크!
        }

        public Integer pop() {
            int[] stampHolder = new int[1];
            Node oldHead;
            Node newHead;
            do {
                oldHead = head.get(stampHolder);
                if (oldHead == null) return null;
                int stamp = stampHolder[0];
                newHead = oldHead.next;
            } while (!head.compareAndSet(
                oldHead, newHead, stamp, stamp + 1));
            return oldHead.value;
        }
    }
}
```

### 7.3 고성능 Concurrent 자료구조

```java
public class ConcurrentDataStructures {

    // 1. ConcurrentHashMap: Lock Striping
    public static void concurrentHashMapExample() {
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();

        // 내부 구조:
        // Segment 0 | Segment 1 | Segment 2 | ... | Segment 15
        //   ↓lock       ↓lock        ↓lock              ↓lock
        // 각 Segment가 독립적인 락을 가짐
        // → 16개 스레드가 동시에 쓰기 가능!

        map.put("key1", 1);
        map.put("key2", 2);

        // Java 8+: CAS 기반으로 더 최적화
        map.computeIfAbsent("key3", k -> {
            // 복잡한 계산
            return 3;
        });
    }

    // 2. ConcurrentLinkedQueue: Lock-Free Queue
    public static void concurrentQueueExample() {
        ConcurrentLinkedQueue<Integer> queue = new ConcurrentLinkedQueue<>();

        // offer: Lock-Free로 추가
        queue.offer(1);
        queue.offer(2);

        // poll: Lock-Free로 제거
        Integer value = queue.poll();

        // CAS 기반으로 구현됨
        // → 매우 높은 동시성 성능!
    }

    // 3. CopyOnWriteArrayList: 읽기 최적화
    public static void copyOnWriteExample() {
        CopyOnWriteArrayList<String> list = new CopyOnWriteArrayList<>();

        // 쓰기: 전체 배열 복사
        list.add("A");  // 내부 배열 복사
        list.add("B");  // 다시 복사
        // → 쓰기는 느림

        // 읽기: Lock 없음, 매우 빠름
        for (String s : list) {
            System.out.println(s);  // Lock 없이 읽기
        }

        // 용도: 읽기가 매우 많고 쓰기가 드문 경우
        // 예: 설정값, 리스너 목록 등
    }

    // 4. LinkedBlockingQueue vs ArrayBlockingQueue
    public static void blockingQueueComparison() {
        // LinkedBlockingQueue: 무제한 크기, 2개 락 (put/take)
        LinkedBlockingQueue<Integer> linkedQueue = new LinkedBlockingQueue<>();

        // ArrayBlockingQueue: 고정 크기, 1개 락 (공유)
        ArrayBlockingQueue<Integer> arrayQueue = new ArrayBlockingQueue<>(100);

        // 성능:
        // - LinkedBlockingQueue: put/take 동시 실행 가능
        // - ArrayBlockingQueue: 캐시 친화적, 메모리 효율적
    }

    // 성능 비교 벤치마크
    public static void benchmark() throws InterruptedException {
        int operations = 1_000_000;
        int threads = 8;

        // 1. ConcurrentHashMap
        ConcurrentHashMap<Integer, Integer> chm = new ConcurrentHashMap<>();
        long chmTime = measureTime(() -> {
            for (int i = 0; i < operations; i++) {
                chm.put(i, i);
            }
        }, threads);
        System.out.println("ConcurrentHashMap: " + chmTime + "ms");

        // 2. Collections.synchronizedMap
        Map<Integer, Integer> syncMap =
            Collections.synchronizedMap(new HashMap<>());
        long syncTime = measureTime(() -> {
            for (int i = 0; i < operations; i++) {
                syncMap.put(i, i);
            }
        }, threads);
        System.out.println("SynchronizedMap: " + syncTime + "ms");

        // 결과:
        // ConcurrentHashMap: 300ms
        // SynchronizedMap: 1200ms (4배 느림!)
    }

    private static long measureTime(Runnable task, int threads)
            throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(threads);
        long start = System.currentTimeMillis();

        for (int i = 0; i < threads; i++) {
            new Thread(() -> {
                task.run();
                latch.countDown();
            }).start();
        }

        latch.await();
        return System.currentTimeMillis() - start;
    }
}
```

---

## 8. Virtual Thread (Java 21+)

### 8.1 전통적인 Platform Thread의 한계

**문제점:**

```
Platform Thread (전통적인 Java 스레드)
= 1:1 매핑 → OS 스레드
→ 생성 비용: 1ms
→ 메모리: 1MB per thread
→ 컨텍스트 스위칭 비용

100,000개 스레드 = 100GB 메모리!
→ 불가능!
```

**C10K 문제:**
- 동시에 10,000개의 연결을 처리하기 어려움
- 스레드 당 1MB → 10GB 메모리

### 8.2 Virtual Thread의 혁신

**개념:**

```
Virtual Thread (경량 스레드)
= N:M 매핑 → 적은 수의 OS 스레드에 많은 Virtual Thread를 매핑
→ 생성 비용: 거의 0
→ 메모리: 수백 bytes
→ JVM이 스케줄링

1,000,000개 Virtual Thread = 수백 MB!
→ 가능!
```

**구조:**

```
┌──────────────────────────────────────────┐
│         Virtual Threads (100만개)        │
│  VT1  VT2  VT3  VT4  ...  VT1000000     │
└──────────────┬───────────────────────────┘
               │ JVM Scheduler
               ↓
┌──────────────────────────────────────────┐
│       Carrier Threads (8개)              │
│  Platform Thread 1, 2, 3, ... 8         │
└──────────────┬───────────────────────────┘
               │ 1:1 Mapping
               ↓
┌──────────────────────────────────────────┐
│         OS Threads (8개)                 │
└──────────────────────────────────────────┘
```

### 8.3 Virtual Thread 사용법

```java
public class VirtualThreadExample {

    // 1. Virtual Thread 생성 (Java 21+)
    public static void basicExample() throws InterruptedException {
        // Platform Thread (전통적인 방식)
        Thread platformThread = new Thread(() -> {
            System.out.println("Platform Thread: " +
                Thread.currentThread());
        });
        platformThread.start();

        // Virtual Thread (새로운 방식)
        Thread virtualThread = Thread.startVirtualThread(() -> {
            System.out.println("Virtual Thread: " +
                Thread.currentThread());
        });

        platformThread.join();
        virtualThread.join();

        // 출력:
        // Platform Thread: Thread[Thread-0,5,main]
        // Virtual Thread: VirtualThread[#21]/runnable@ForkJoinPool-1-worker-1
    }

    // 2. 100만개 Virtual Thread 생성
    public static void millionThreadsExample() throws InterruptedException {
        long start = System.currentTimeMillis();

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < 1_000_000; i++) {
                executor.submit(() -> {
                    try {
                        Thread.sleep(1000);  // I/O 시뮬레이션
                    } catch (InterruptedException e) {}
                    return "done";
                });
            }
        }  // 자동으로 shutdown & awaitTermination

        long end = System.currentTimeMillis();
        System.out.println("100만개 Virtual Thread: " + (end - start) + "ms");
        // 결과: 약 1200ms (1.2초!)

        // Platform Thread로 같은 작업 시도하면?
        // → OutOfMemoryError!
    }

    // 3. 성능 비교: Platform vs Virtual
    public static void performanceComparison() throws InterruptedException {
        int taskCount = 10_000;

        // Platform Thread Pool
        long start1 = System.currentTimeMillis();
        try (var executor = Executors.newFixedThreadPool(200)) {
            for (int i = 0; i < taskCount; i++) {
                executor.submit(() -> {
                    try {
                        Thread.sleep(100);  // I/O 시뮬레이션
                    } catch (InterruptedException e) {}
                });
            }
        }
        long end1 = System.currentTimeMillis();
        System.out.println("Platform Thread Pool (200): " +
            (end1 - start1) + "ms");
        // 결과: 약 5000ms

        // Virtual Thread
        long start2 = System.currentTimeMillis();
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < taskCount; i++) {
                executor.submit(() -> {
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException e) {}
                });
            }
        }
        long end2 = System.currentTimeMillis();
        System.out.println("Virtual Thread: " + (end2 - start2) + "ms");
        // 결과: 약 150ms (30배 빨라짐!)
    }

    // 4. Spring Boot에서 Virtual Thread 사용
    /*
    @Configuration
    public class VirtualThreadConfig {

        @Bean
        public TomcatProtocolHandlerCustomizer<?> protocolHandlerVirtualThreadExecutorCustomizer() {
            return protocolHandler -> {
                protocolHandler.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
            };
        }
    }

    // application.properties
    // spring.threads.virtual.enabled=true
    */
}
```

### 8.4 Virtual Thread 주의사항

```java
public class VirtualThreadPitfalls {

    // ⚠️ 주의 1: synchronized는 Pinning 발생
    private static final Object lock = new Object();

    public static void pinningProblem() {
        Thread.startVirtualThread(() -> {
            synchronized (lock) {  // ⚠️ Carrier Thread에 고정됨!
                try {
                    Thread.sleep(1000);  // 다른 Virtual Thread로 전환 안 됨
                } catch (InterruptedException e) {}
            }
        });

        // ✅ 해결: ReentrantLock 사용
        var betterLock = new ReentrantLock();
        Thread.startVirtualThread(() -> {
            betterLock.lock();
            try {
                Thread.sleep(1000);  // 다른 Virtual Thread로 전환 가능!
            } catch (InterruptedException e) {
            } finally {
                betterLock.unlock();
            }
        });
    }

    // ⚠️ 주의 2: CPU-Bound 작업에는 부적합
    public static void cpuBoundProblem() {
        // CPU-Bound 작업
        Runnable cpuIntensive = () -> {
            long sum = 0;
            for (long i = 0; i < 1_000_000_000L; i++) {
                sum += i;
            }
        };

        // Virtual Thread (비효율적)
        long start1 = System.currentTimeMillis();
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < 100; i++) {
                executor.submit(cpuIntensive);
            }
        }
        long end1 = System.currentTimeMillis();
        System.out.println("Virtual Thread (CPU-Bound): " +
            (end1 - start1) + "ms");

        // Platform Thread Pool (효율적)
        int cores = Runtime.getRuntime().availableProcessors();
        long start2 = System.currentTimeMillis();
        try (var executor = Executors.newFixedThreadPool(cores)) {
            for (int i = 0; i < 100; i++) {
                executor.submit(cpuIntensive);
            }
        }
        long end2 = System.currentTimeMillis();
        System.out.println("Platform Thread Pool (CPU-Bound): " +
            (end2 - start2) + "ms");

        // Virtual Thread가 더 느림!
        // → CPU-Bound는 Platform Thread 사용
    }

    // ✅ Virtual Thread 최적 사용 사례
    public static void bestUseCase() {
        // I/O-Bound 작업에 최적
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            // 수백만 개의 HTTP 요청 처리
            for (int i = 0; i < 1_000_000; i++) {
                executor.submit(() -> {
                    // DB 쿼리
                    // HTTP API 호출
                    // 파일 읽기/쓰기
                    // → I/O 대기 중 다른 Virtual Thread 실행!
                });
            }
        }
    }
}
```

**Virtual Thread 선택 가이드:**

| 작업 유형 | Platform Thread | Virtual Thread |
|----------|----------------|---------------|
| CPU-Bound | ✅ 권장 (코어 수만큼) | ❌ 비효율적 |
| I/O-Bound (적은 요청) | ✅ 가능 | ✅ 가능 |
| I/O-Bound (많은 요청) | ❌ 메모리 부족 | ✅ 최적 |
| synchronized 많이 사용 | ✅ 가능 | ⚠️ Pinning 주의 |
| 레거시 코드 | ✅ 호환성 좋음 | ⚠️ 테스트 필요 |

---

## 9. 성능 측정과 프로파일링

### 9.1 JMH (Java Microbenchmark Harness)

**정확한 벤치마크 도구:**

```java
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(1)
public class ThreadBenchmark {

    private AtomicInteger atomicCounter;
    private int volatileCounter;
    private int plainCounter;

    @Setup
    public void setup() {
        atomicCounter = new AtomicInteger(0);
        volatileCounter = 0;
        plainCounter = 0;
    }

    @Benchmark
    public int atomicIncrement() {
        return atomicCounter.incrementAndGet();
    }

    @Benchmark
    public int volatileIncrement() {
        return ++volatileCounter;  // 원자성 보장 안 됨!
    }

    @Benchmark
    public int plainIncrement() {
        return ++plainCounter;
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }

    // 결과 예시:
    // Benchmark                          Mode  Cnt   Score   Error  Units
    // ThreadBenchmark.atomicIncrement    avgt   10  15.234 ± 0.321  ns/op
    // ThreadBenchmark.volatileIncrement  avgt   10   8.123 ± 0.156  ns/op
    // ThreadBenchmark.plainIncrement     avgt   10   2.456 ± 0.087  ns/op
    //
    // AtomicInteger가 가장 느리지만 동시성 안전!
}
```

### 9.2 스레드 덤프 분석

```java
public class ThreadDumpAnalysis {

    public static void generateThreadDump() {
        ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
        ThreadInfo[] threadInfos = threadMXBean.dumpAllThreads(true, true);

        for (ThreadInfo info : threadInfos) {
            System.out.println("Thread: " + info.getThreadName());
            System.out.println("  State: " + info.getThreadState());
            System.out.println("  CPU Time: " +
                threadMXBean.getThreadCpuTime(info.getThreadId()) / 1_000_000 + "ms");

            // 락 정보
            if (info.getLockName() != null) {
                System.out.println("  Waiting on: " + info.getLockName());
            }

            // 스택 트레이스
            for (StackTraceElement ste : info.getStackTrace()) {
                System.out.println("    " + ste);
            }

            // Deadlock 감지
            long[] deadlockedThreads = threadMXBean.findDeadlockedThreads();
            if (deadlockedThreads != null) {
                System.out.println("⚠️ Deadlock detected!");
                for (long tid : deadlockedThreads) {
                    ThreadInfo deadlocked = threadMXBean.getThreadInfo(tid);
                    System.out.println("  Deadlocked: " +
                        deadlocked.getThreadName());
                }
            }
        }
    }

    // Deadlock 예제
    public static void deadlockExample() {
        Object lock1 = new Object();
        Object lock2 = new Object();

        Thread t1 = new Thread(() -> {
            synchronized (lock1) {
                System.out.println("T1: holding lock1");
                try { Thread.sleep(100); } catch (InterruptedException e) {}
                System.out.println("T1: waiting for lock2");
                synchronized (lock2) {
                    System.out.println("T1: acquired lock2");
                }
            }
        });

        Thread t2 = new Thread(() -> {
            synchronized (lock2) {
                System.out.println("T2: holding lock2");
                try { Thread.sleep(100); } catch (InterruptedException e) {}
                System.out.println("T2: waiting for lock1");
                synchronized (lock1) {
                    System.out.println("T2: acquired lock1");
                }
            }
        });

        t1.start();
        t2.start();

        // Deadlock 발생!
        // jstack <pid> 또는 jconsole로 확인 가능
    }
}
```

### 9.3 VisualVM / JProfiler 프로파일링

```java
public class ProfilingExample {

    // CPU 프로파일링 대상
    public static void cpuIntensiveTask() {
        long sum = 0;
        for (int i = 0; i < 100_000_000; i++) {
            sum += Math.sqrt(i);
        }
    }

    // 메모리 프로파일링 대상
    public static void memoryIntensiveTask() {
        List<byte[]> list = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            list.add(new byte[1024 * 1024]);  // 1MB씩 할당
        }
    }

    // 스레드 경합 프로파일링 대상
    private static int counter = 0;
    private static final Object lock = new Object();

    public static void contentionTask() throws InterruptedException {
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 100_000; j++) {
                    synchronized (lock) {  // 경합 발생!
                        counter++;
                    }
                }
            });
            threads[i].start();
        }

        for (Thread t : threads) {
            t.join();
        }
    }

    public static void main(String[] args) throws InterruptedException {
        // VisualVM으로 모니터링:
        // 1. CPU Sampler: CPU 사용률 높은 메서드 찾기
        // 2. Memory Sampler: 메모리 누수 찾기
        // 3. Threads: 스레드 상태, 경합 확인

        cpuIntensiveTask();
        memoryIntensiveTask();
        contentionTask();
    }
}
```

### 9.4 성능 메트릭 수집

```java
@Component
public class ThreadPoolMetrics {

    private final MeterRegistry meterRegistry;
    private final ThreadPoolTaskExecutor executor;

    public ThreadPoolMetrics(MeterRegistry meterRegistry,
                            ThreadPoolTaskExecutor executor) {
        this.meterRegistry = meterRegistry;
        this.executor = executor;

        // Micrometer로 메트릭 등록
        Gauge.builder("thread.pool.active", executor, ThreadPoolTaskExecutor::getActiveCount)
            .description("Active thread count")
            .register(meterRegistry);

        Gauge.builder("thread.pool.size", executor, ThreadPoolTaskExecutor::getPoolSize)
            .description("Current pool size")
            .register(meterRegistry);

        Gauge.builder("thread.pool.queue.size", executor,
            e -> e.getThreadPoolExecutor().getQueue().size())
            .description("Queue size")
            .register(meterRegistry);

        Gauge.builder("thread.pool.max", executor, ThreadPoolTaskExecutor::getMaxPoolSize)
            .description("Max pool size")
            .register(meterRegistry);
    }

    // 실시간 모니터링
    @Scheduled(fixedRate = 5000)
    public void logMetrics() {
        ThreadPoolExecutor pool = executor.getThreadPoolExecutor();

        log.info("Thread Pool Metrics:");
        log.info("  Active: {}", pool.getActiveCount());
        log.info("  Pool Size: {}", pool.getPoolSize());
        log.info("  Queue: {}/{}", pool.getQueue().size(),
            ((LinkedBlockingQueue<?>) pool.getQueue()).remainingCapacity());
        log.info("  Completed: {}", pool.getCompletedTaskCount());
        log.info("  Total: {}", pool.getTaskCount());

        // 경고 조건
        if (pool.getActiveCount() == pool.getMaximumPoolSize()) {
            log.warn("⚠️ Thread pool is fully utilized!");
        }

        if (pool.getQueue().size() > pool.getQueue().size() * 0.8) {
            log.warn("⚠️ Queue is 80% full!");
        }
    }
}
```

---

## 10. 실무 성능 최적화 패턴

### 10.1 패턴 1: 작업 분할 (Fork/Join)

```java
public class ForkJoinExample {

    // 큰 작업을 작은 작업으로 분할
    static class SumTask extends RecursiveTask<Long> {
        private static final int THRESHOLD = 10_000;
        private final long[] array;
        private final int start;
        private final int end;

        public SumTask(long[] array, int start, int end) {
            this.array = array;
            this.start = start;
            this.end = end;
        }

        @Override
        protected Long compute() {
            int length = end - start;

            // 작은 작업: 직접 계산
            if (length <= THRESHOLD) {
                long sum = 0;
                for (int i = start; i < end; i++) {
                    sum += array[i];
                }
                return sum;
            }

            // 큰 작업: 분할 정복
            int mid = start + length / 2;
            SumTask leftTask = new SumTask(array, start, mid);
            SumTask rightTask = new SumTask(array, mid, end);

            leftTask.fork();  // 비동기 실행
            long rightResult = rightTask.compute();  // 동기 실행
            long leftResult = leftTask.join();  // 결과 대기

            return leftResult + rightResult;
        }
    }

    public static void benchmark() {
        long[] array = new long[100_000_000];
        for (int i = 0; i < array.length; i++) {
            array[i] = i;
        }

        // 1. 단일 스레드
        long start1 = System.currentTimeMillis();
        long sum1 = 0;
        for (long value : array) {
            sum1 += value;
        }
        long end1 = System.currentTimeMillis();
        System.out.println("Single Thread: " + (end1 - start1) + "ms");

        // 2. Fork/Join
        ForkJoinPool pool = ForkJoinPool.commonPool();
        long start2 = System.currentTimeMillis();
        long sum2 = pool.invoke(new SumTask(array, 0, array.length));
        long end2 = System.currentTimeMillis();
        System.out.println("Fork/Join: " + (end2 - start2) + "ms");

        // 결과:
        // Single Thread: 200ms
        // Fork/Join: 60ms (약 3.3배 빨라짐)
    }
}
```

### 10.2 패턴 2: 배치 처리

```java
@Service
public class BatchProcessingService {

    private final BlockingQueue<Task> taskQueue = new LinkedBlockingQueue<>();
    private final ScheduledExecutorService batchProcessor =
        Executors.newSingleThreadScheduledExecutor();

    @PostConstruct
    public void init() {
        // 100ms마다 배치 처리
        batchProcessor.scheduleAtFixedRate(
            this::processBatch, 0, 100, TimeUnit.MILLISECONDS);
    }

    // ❌ 나쁜 예: 각 요청마다 DB 쿼리
    public void processImmediately(Task task) {
        // DB에 즉시 저장
        taskRepository.save(task);  // 1개씩 저장 → 느림!
    }

    // ✅ 좋은 예: 배치로 모아서 처리
    public void processInBatch(Task task) {
        taskQueue.offer(task);  // 큐에 추가만
    }

    private void processBatch() {
        List<Task> batch = new ArrayList<>();
        taskQueue.drainTo(batch, 1000);  // 최대 1000개씩

        if (!batch.isEmpty()) {
            // 한 번에 저장
            taskRepository.saveAll(batch);  // Bulk Insert → 빠름!
            log.info("Processed {} tasks", batch.size());
        }
    }

    // 성능 비교:
    // processImmediately: 10,000개 → 30초
    // processInBatch: 10,000개 → 2초 (15배 빨라짐!)
}
```

### 10.3 패턴 3: 비동기 프로그래밍

```java
@Service
public class AsyncPatternService {

    // ❌ 나쁜 예: 동기 처리
    public OrderResult processSynchronous(Order order) {
        // 1. 재고 확인 (300ms)
        boolean stockAvailable = checkStock(order);

        // 2. 결제 처리 (500ms)
        boolean paymentSuccess = processPayment(order);

        // 3. 배송 예약 (200ms)
        boolean shippingReserved = reserveShipping(order);

        // 총 시간: 300 + 500 + 200 = 1000ms
        return new OrderResult(stockAvailable, paymentSuccess, shippingReserved);
    }

    // ✅ 좋은 예: 비동기 처리
    public CompletableFuture<OrderResult> processAsynchronous(Order order) {
        // 1, 2, 3을 동시에 실행
        CompletableFuture<Boolean> stockFuture =
            CompletableFuture.supplyAsync(() -> checkStock(order));

        CompletableFuture<Boolean> paymentFuture =
            CompletableFuture.supplyAsync(() -> processPayment(order));

        CompletableFuture<Boolean> shippingFuture =
            CompletableFuture.supplyAsync(() -> reserveShipping(order));

        // 모든 작업 완료 대기
        return CompletableFuture.allOf(stockFuture, paymentFuture, shippingFuture)
            .thenApply(v -> new OrderResult(
                stockFuture.join(),
                paymentFuture.join(),
                shippingFuture.join()
            ));

        // 총 시간: max(300, 500, 200) = 500ms (2배 빨라짐!)
    }

    // 더 복잡한 예: 순차 + 병렬
    public CompletableFuture<Order> processComplex(Order order) {
        return CompletableFuture.supplyAsync(() -> {
            // 1. 주문 검증 (순차 처리 필요)
            validateOrder(order);
            return order;
        })
        .thenCompose(validOrder -> {
            // 2. 재고 & 결제 병렬 처리
            CompletableFuture<Void> stockFuture =
                CompletableFuture.runAsync(() -> checkStock(validOrder));
            CompletableFuture<Void> paymentFuture =
                CompletableFuture.runAsync(() -> processPayment(validOrder));

            return CompletableFuture.allOf(stockFuture, paymentFuture)
                .thenApply(v -> validOrder);
        })
        .thenApply(processedOrder -> {
            // 3. 배송 예약 (마지막 단계)
            reserveShipping(processedOrder);
            return processedOrder;
        })
        .exceptionally(ex -> {
            // 에러 처리
            log.error("Order processing failed", ex);
            return null;
        });
    }
}
```

### 10.4 패턴 4: 캐시와 스레드 로컬

```java
public class ThreadLocalCachePattern {

    // ❌ 나쁜 예: 스레드 간 경합
    private static final SimpleDateFormat DATE_FORMAT =
        new SimpleDateFormat("yyyy-MM-dd");  // Not thread-safe!

    public static String formatDateBad(Date date) {
        synchronized (DATE_FORMAT) {  // 모든 스레드가 대기
            return DATE_FORMAT.format(date);
        }
    }

    // ✅ 좋은 예: ThreadLocal 사용
    private static final ThreadLocal<SimpleDateFormat> DATE_FORMAT_TL =
        ThreadLocal.withInitial(() -> new SimpleDateFormat("yyyy-MM-dd"));

    public static String formatDateGood(Date date) {
        return DATE_FORMAT_TL.get().format(date);  // Lock 없음!
    }

    // ThreadLocal 캐시 패턴
    static class ThreadLocalCache<K, V> {
        private final ThreadLocal<Map<K, V>> cache =
            ThreadLocal.withInitial(HashMap::new);

        public V get(K key, Function<K, V> loader) {
            Map<K, V> localCache = cache.get();
            return localCache.computeIfAbsent(key, loader);
        }

        public void clear() {
            cache.remove();  // 메모리 누수 방지!
        }
    }

    // 사용 예시
    public static void threadLocalCacheExample() {
        ThreadLocalCache<String, User> userCache = new ThreadLocalCache<>();

        // 각 스레드가 독립적인 캐시 사용
        User user = userCache.get("user1", userId -> {
            // DB에서 로드 (무거운 작업)
            return userRepository.findById(userId);
        });

        // 같은 스레드에서 다시 요청 → 캐시에서 바로 반환
        User cachedUser = userCache.get("user1", userId -> {
            // 호출 안 됨!
            return null;
        });

        // 작업 완료 후 반드시 정리!
        userCache.clear();
    }

    // 성능 비교:
    // synchronized: 10,000 calls → 500ms
    // ThreadLocal: 10,000 calls → 50ms (10배 빨라짐!)
}
```

### 10.5 패턴 5: Reactive Programming

```java
// Project Reactor 사용 예시
@Service
public class ReactivePatternService {

    // ❌ 전통적인 방식
    public List<User> getUsersBlocking(List<Long> userIds) {
        List<User> users = new ArrayList<>();
        for (Long id : userIds) {
            User user = restTemplate.getForObject(
                "https://api.example.com/users/" + id, User.class);
            users.add(user);
        }
        // 순차 처리: 100개 × 100ms = 10초
        return users;
    }

    // ✅ Reactive 방식
    public Flux<User> getUsersReactive(List<Long> userIds) {
        return Flux.fromIterable(userIds)
            .flatMap(id -> webClient
                .get()
                .uri("/users/{id}", id)
                .retrieve()
                .bodyToMono(User.class)
            )
            .onErrorResume(error -> {
                log.error("Failed to fetch user", error);
                return Mono.empty();
            });
        // 병렬 처리: max(100ms) = 100ms (100배 빨라짐!)
    }

    // Backpressure 처리
    public Flux<ProcessedData> processWithBackpressure(Flux<Data> dataStream) {
        return dataStream
            .buffer(100)  // 100개씩 배치
            .flatMap(batch -> processBatch(batch), 4)  // 최대 4개 배치 동시 처리
            .onBackpressureBuffer(1000);  // 1000개까지 버퍼링
    }
}
```

### 10.6 실무 체크리스트

```java
/**
 * 스레드 성능 최적화 체크리스트
 */
public class PerformanceChecklist {

    // ✅ 1. 스레드 풀 크기 적절한가?
    // - CPU-Bound: 코어 수 + 1
    // - I/O-Bound: 코어 수 × (1 + Wait/Service Time)
    private int calculateOptimalPoolSize() {
        int cores = Runtime.getRuntime().availableProcessors();
        double cpuUtilization = 0.8;  // 목표 CPU 사용률
        double waitTimeRatio = 0.9;   // Wait Time / Service Time

        return (int) (cores / cpuUtilization * (1 + waitTimeRatio));
    }

    // ✅ 2. 스레드 풀 모니터링하는가?
    // - Active threads
    // - Queue size
    // - Rejected tasks
    // - Completed tasks

    // ✅ 3. 적절한 동시성 제어 사용하는가?
    // - AtomicInteger: 단순 카운터
    // - ConcurrentHashMap: 공유 Map
    // - synchronized: 복잡한 비즈니스 로직
    // - DB Lock: 데이터 정합성

    // ✅ 4. 불필요한 동기화 피하는가?
    // - Lock 범위 최소화
    // - 읽기 전용은 Lock 불필요
    // - ThreadLocal 활용

    // ✅ 5. Deadlock 방지하는가?
    // - Lock 순서 일관성
    // - Timeout 설정
    // - tryLock 사용

    // ✅ 6. 스레드 안전성 문서화하는가?
    /**
     * @ThreadSafe
     * 이 클래스는 스레드 안전합니다.
     */
    public class ThreadSafeClass {
        private final AtomicInteger count = new AtomicInteger(0);

        public int increment() {
            return count.incrementAndGet();
        }
    }

    /**
     * @NotThreadSafe
     * 이 클래스는 스레드 안전하지 않습니다.
     * 외부에서 동기화 필요!
     */
    public class NotThreadSafeClass {
        private int count = 0;

        public int increment() {
            return ++count;  // Race Condition!
        }
    }

    // ✅ 7. 테스트 코드 작성했는가?
    // - 동시성 테스트
    // - 부하 테스트
    // - Deadlock 테스트

    // ✅ 8. 성능 프로파일링 했는가?
    // - CPU 사용률
    // - 메모리 사용량
    // - 스레드 경합
    // - GC 부담
}
```

---

## 마치며: 스레드 마스터가 되기 위한 로드맵

### 단계별 학습

**Level 1: 기초 (1-2개월)**
- ✅ 프로세스 vs 스레드 이해
- ✅ Thread 생성과 실행
- ✅ synchronized 사용법
- ✅ wait/notify 이해

**Level 2: 중급 (2-3개월)**
- ✅ ExecutorService 사용
- ✅ Concurrent Collections
- ✅ AtomicInteger, volatile
- ✅ 동시성 테스트 작성

**Level 3: 고급 (3-6개월)**
- ✅ ThreadPoolExecutor 설정
- ✅ Lock, ReentrantLock
- ✅ CompletableFuture
- ✅ 성능 프로파일링

**Level 4: 전문가 (6개월+)**
- ✅ Java Memory Model 이해
- ✅ Lock-Free 알고리즘
- ✅ Virtual Thread 활용
- ✅ 아키텍처 레벨 설계

### 추천 자료

**책:**
1. "Java Concurrency in Practice" - Brian Goetz (필독!)
2. "Operating System Concepts" - Silberschatz (OS 이론)
3. "The Art of Multiprocessor Programming" - Maurice Herlihy

**온라인 강의:**
1. Udemy: "Java Multithreading, Concurrency & Performance Optimization"
2. Coursera: "Parallel, Concurrent, and Distributed Programming in Java"

**블로그/아티클:**
1. Oracle Java Tutorials - Concurrency
2. Baeldung - Java Concurrency Series
3. DZone - Java Zone

**실습:**
1. 직접 동시성 버그 만들고 해결하기
2. 스레드 풀 설정 실험하기
3. 성능 벤치마크 작성하기
4. 오픈소스 코드 읽기 (Netty, Tomcat 등)

### 핵심 원칙

```
1. "측정할 수 없으면 개선할 수 없다"
   → 항상 측정하고 프로파일링하라

2. "조기 최적화는 모든 악의 근원"
   → 필요할 때만 최적화하라

3. "동시성 버그는 재현하기 어렵다"
   → 테스트 코드로 미리 잡아라

4. "단순함이 최고의 최적화다"
   → 복잡한 코드보다 단순하고 명확한 코드

5. "스레드는 비싸다"
   → 스레드 풀을 사용하라

6. "공유는 위험하다"
   → 가능하면 공유를 피하고, 불가피하면 보호하라

7. "I/O는 병렬화하라, CPU는 신중하게"
   → 작업 유형에 맞는 전략 선택
```

---

**작성 완료!** 🎉

이 문서가 스레드에 대한 깊은 이해와 성능 향상에 도움이 되길 바랍니다!

질문이나 피드백은 언제든 환영합니다.

**Good Luck with Your Threading Journey!** 🚀
