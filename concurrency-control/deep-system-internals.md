# 시스템 내부 Deep Dive
## OS 커널과 DB 엔진 개발자 관점

> **목표**: "만약 내가 OS나 DB를 직접 개발한다면?" 관점으로 시스템 내부를 이해하기

---

## 목차

1. [커널은 스레드를 어떻게 관리하는가?](#1-커널은-스레드를-어떻게-관리하는가)
2. [스레드는 프로세스당 몇 개까지?](#2-스레드는-프로세스당-몇-개까지)
3. [스레드/프로세스 생명주기와 중요성](#3-스레드프로세스-생명주기와-중요성)
4. [DB는 비관적 락을 어떻게 구현하는가?](#4-db는-비관적-락을-어떻게-구현하는가)

---

# 1. 커널은 스레드를 어떻게 관리하는가?

## 1-1. "내가 OS 개발자라면?" - 필요한 자료구조

### 핵심 자료구조: TCB (Thread Control Block)

스레드의 "주민등록증"이다. 커널은 각 스레드의 모든 정보를 TCB에 저장한다.

```c
// 커널 메모리에 존재하는 스레드 정보
struct thread_control_block {
    // 1. 신원 정보
    int thread_id;                    // 스레드 고유 ID (TID)
    pid_t process_id;                 // 소속 프로세스 ID

    // 2. 실행 상태 정보
    enum thread_state state;          // NEW, READY, RUNNING, BLOCKED, TERMINATED
    int priority;                     // 우선순위 (스케줄링용)

    // 3. CPU 실행 컨텍스트 (핵심!)
    struct cpu_context {
        unsigned long pc;             // Program Counter (다음 실행할 명령어 주소)
        unsigned long sp;             // Stack Pointer (스택 최상단 주소)
        unsigned long registers[32];  // 범용 레지스터 값들 (작업 내용)
    } context;

    // 4. 메모리 정보
    void* stack_base;                 // 스택 시작 주소
    size_t stack_size;                // 스택 크기 (보통 1MB)

    // 5. 스케줄링 정보
    unsigned long long cpu_time_used; // 누적 CPU 사용 시간
    struct timespec create_time;      // 생성 시각

    // 6. 동기화 정보
    void* waiting_on;                 // 어떤 락/이벤트를 기다리는지

    // 7. 링크드 리스트 연결
    struct thread_control_block* next; // 다음 TCB (큐 구현용)
};

typedef struct thread_control_block TCB;
```

**왜 이런 정보가 필요한가?**
- `context`: 컨텍스트 스위칭 시 작업 내용을 백업/복원
- `state`: 스케줄러가 어느 큐에 넣을지 결정
- `stack_base`: 스레드마다 독립적인 스택 공간
- `waiting_on`: 왜 Blocked 상태인지 파악

---

## 1-2. 스케줄러 로직 - 스레드 생명의 지배자

### 핵심 자료구조: Ready Queue & Wait Queue

```c
// 전역 큐들 (커널 메모리에 존재)
TCB* ready_queue_head = NULL;   // 실행 대기 중인 스레드들
TCB* wait_queue_head = NULL;    // Blocked 상태 스레드들
TCB* current_thread = NULL;     // 현재 CPU에서 실행 중인 스레드
```

---

### 스레드 생성: `create_thread()`

```c
// Java: new Thread().start()
// ↓ JNI 호출
// ↓ System Call
// ↓ 커널의 이 함수 실행!

TCB* create_thread(void* (*start_function)(void*), void* arg) {
    // 1. TCB 메모리 할당 (커널 힙)
    TCB* new_tcb = (TCB*)kmalloc(sizeof(TCB));

    // 2. 신원 정보 설정
    new_tcb->thread_id = generate_unique_tid();
    new_tcb->process_id = current_process->pid;

    // 3. 스택 공간 할당 (~1MB)
    new_tcb->stack_size = 1024 * 1024;  // 1MB
    new_tcb->stack_base = (void*)kmalloc(new_tcb->stack_size);

    // 4. CPU 컨텍스트 초기화
    new_tcb->context.pc = (unsigned long)start_function;  // 시작 함수 주소
    new_tcb->context.sp = (unsigned long)(new_tcb->stack_base + new_tcb->stack_size);
    memset(new_tcb->context.registers, 0, sizeof(new_tcb->context.registers));

    // 5. 상태 설정
    new_tcb->state = READY;  // 실행 대기
    new_tcb->priority = DEFAULT_PRIORITY;

    // 6. Ready Queue에 추가
    enqueue_ready(new_tcb);

    // 7. 스케줄러에게 알림 (새 스레드가 생겼어요!)
    schedule();

    return new_tcb;
}

// Ready Queue에 추가 (단순 링크드 리스트)
void enqueue_ready(TCB* tcb) {
    tcb->next = NULL;
    if (ready_queue_head == NULL) {
        ready_queue_head = tcb;
    } else {
        // 꼬리에 추가
        TCB* tail = ready_queue_head;
        while (tail->next != NULL) {
            tail = tail->next;
        }
        tail->next = tcb;
    }
}
```

**비용 분석:**
- TCB 할당: ~1KB (구조체 크기)
- Stack 할당: ~1MB (스레드당)
- **총 메모리:** 약 1MB + α
- **시간:** ~1ms (메모리 할당 + 초기화)

---

### 컨텍스트 스위칭: `context_switch()`

**이게 동시성의 마법이다!**

```c
// 타이머 인터럽트 (보통 10ms마다) 또는 스레드가 Blocked될 때 호출
void context_switch(TCB* old_thread, TCB* new_thread) {
    // === PHASE 1: SAVE (현재 스레드 백업) ===
    // CPU 레지스터 → TCB로 저장
    asm volatile (
        "mov %%rax, %0\n"
        "mov %%rbx, %1\n"
        "mov %%rcx, %2\n"
        // ... 32개 레지스터 모두 저장
        : "=m"(old_thread->context.registers[0]),
          "=m"(old_thread->context.registers[1]),
          "=m"(old_thread->context.registers[2])
    );

    old_thread->context.pc = read_instruction_pointer();
    old_thread->context.sp = read_stack_pointer();

    // === PHASE 2: LOAD (새 스레드 복원) ===
    // TCB → CPU 레지스터로 복원
    asm volatile (
        "mov %0, %%rax\n"
        "mov %1, %%rbx\n"
        "mov %2, %%rcx\n"
        // ... 32개 레지스터 모두 복원
        :
        : "m"(new_thread->context.registers[0]),
          "m"(new_thread->context.registers[1]),
          "m"(new_thread->context.registers[2])
    );

    write_instruction_pointer(new_thread->context.pc);
    write_stack_pointer(new_thread->context.sp);

    // === PHASE 3: JUMP ===
    // 새 스레드의 명령어 주소로 점프!
    // CPU는 이제 완전히 다른 스레드를 실행한다
    current_thread = new_thread;

    // 여기서부터는 new_thread의 코드가 실행됨!
}
```

**비용:**
- 직접 비용: 5-10μs (레지스터 저장/복원)
- 간접 비용: 50-100μs (CPU 캐시 미스)
- **총:** 약 100μs

---

### 스케줄러: `schedule()`

**CPU를 누구에게 줄까?**

```c
void schedule(void) {
    // 현재 실행 중인 스레드
    TCB* old_thread = current_thread;

    // Ready Queue에서 다음 스레드 선택
    TCB* new_thread = dequeue_ready();  // 맨 앞에서 꺼냄

    if (new_thread == NULL) {
        // 실행할 스레드가 없음 → Idle 상태
        run_idle_thread();
        return;
    }

    // 상태 변경
    if (old_thread->state == RUNNING) {
        old_thread->state = READY;
        enqueue_ready(old_thread);  // 다시 줄 서기
    }
    new_thread->state = RUNNING;

    // 컨텍스트 스위칭!
    context_switch(old_thread, new_thread);
}

TCB* dequeue_ready(void) {
    if (ready_queue_head == NULL) {
        return NULL;
    }
    TCB* tcb = ready_queue_head;
    ready_queue_head = ready_queue_head->next;
    return tcb;
}
```

---

### Blocked 상태 처리

**"좋아요" 코드에서 락을 기다릴 때 일어나는 일!**

```c
// Thread A가 DB 락 요청 → 이미 점유됨 → Blocked!
void block_current_thread(void* lock_object) {
    TCB* current = current_thread;

    // 1. 상태 변경
    current->state = BLOCKED;
    current->waiting_on = lock_object;  // 뭘 기다리는지 저장

    // 2. Wait Queue로 이동
    enqueue_wait(current);

    // 3. CPU 반납 (다른 스레드에게 양보)
    schedule();  // 이 스레드는 여기서 멈춤!

    // ... (나중에 깨어나면 여기서부터 재개)
}

void enqueue_wait(TCB* tcb) {
    tcb->next = NULL;
    if (wait_queue_head == NULL) {
        wait_queue_head = tcb;
    } else {
        TCB* tail = wait_queue_head;
        while (tail->next != NULL) {
            tail = tail->next;
        }
        tail->next = tcb;
    }
}
```

---

### 스레드 깨우기 (Wake Up)

**락이 해제되면 DB 드라이버가 이 함수를 호출!**

```c
// 락 해제 → OS에게 "이 락 기다리던 스레드 깨워줘!"
void wakeup_threads_waiting_on(void* lock_object) {
    TCB* prev = NULL;
    TCB* current = wait_queue_head;

    // Wait Queue를 순회하며 해당 락을 기다리던 스레드 찾기
    while (current != NULL) {
        if (current->waiting_on == lock_object) {
            // 1. Wait Queue에서 제거
            if (prev == NULL) {
                wait_queue_head = current->next;
            } else {
                prev->next = current->next;
            }

            // 2. 상태 변경
            current->state = READY;
            current->waiting_on = NULL;

            // 3. Ready Queue로 이동
            enqueue_ready(current);

            // 첫 번째만 깨우고 끝 (하나씩 순차 처리)
            break;
        }
        prev = current;
        current = current->next;
    }
}
```

---

## 1-3. "좋아요" 요청에서 커널의 동작

### 시나리오: 1000개 요청이 동시에 들어올 때

```
Thread A:
  [Ready Queue] → [Running: DB 락 획득 성공!]
  → [작업 중...]
  → [커밋 & 락 해제]
  → [Terminated]

Thread B~Z (999개):
  [Ready Queue]
  → [Running: DB 락 요청]
  → DB: "점유 중, 대기하세요"
  → block_current_thread() 호출
  → [Blocked 상태로 Wait Queue 이동]
  → **CPU 소모 없음! (Ready Queue에 없음)**

Thread A 커밋 후:
  DB 드라이버 → wakeup_threads_waiting_on(lock) 호출
  Thread B: [Wait Queue] → [Ready Queue] → [Running] → ...
  (1000번 반복)
```

**핵심:**
- Blocked 상태는 Ready Queue에 없다!
- 스케줄러는 Ready Queue만 본다!
- → CPU 자원 절약!

---

# 2. 스레드는 프로세스당 몇 개까지?

## 2-1. RAM의 물리적 한계

### 계산: 메모리 소모량

**스레드 1개 생성 시 필요한 메모리:**

```
1. Stack 공간 (사용자 영역):
   - Java 기본: 1MB
   - -Xss 옵션으로 조정 가능
   - 예: -Xss512k → 512KB

2. TCB (커널 영역):
   - 구조체 크기: ~1KB
   - 커널 스택: ~8KB (커널 모드 실행용)

총: 약 1MB + 9KB ≈ 1MB (스택이 압도적)
```

### 실제 한계 계산

**시나리오: 서버 메모리 4GB**

```
총 메모리: 4GB
OS 사용: 1GB
JVM Heap: 2GB
기타: 500MB
--------------------
남은 메모리: 500MB

스레드당 메모리: 1MB
최대 스레드 수: 500개
```

**32비트 OS의 경우:**
```
프로세스당 가상 메모리 주소 공간: 2GB (Linux)
스레드당 Stack: 1MB
--------------------
이론적 최대: 2048개

실제 한계: ~1500개 (TCB, 커널 스택 등 고려)
```

**64비트 OS의 경우:**
```
가상 메모리: 이론상 무제한 (128TB)
실제 한계: 물리 RAM
```

---

## 2-2. CPU 코어와 병렬성

### 진정한 병렬성

```
8코어 CPU가 있을 때:

특정 시점 t:
  Core 1: Thread A 실행 중
  Core 2: Thread B 실행 중
  Core 3: Thread C 실행 중
  Core 4: Thread D 실행 중
  Core 5: Thread E 실행 중
  Core 6: Thread F 실행 중
  Core 7: Thread G 실행 중
  Core 8: Thread H 실행 중

→ 정확히 8개만 진짜 "동시" 실행!

Thread I~Z는?
→ Ready Queue에서 대기
→ 컨텍스트 스위칭으로 번갈아 실행
→ "동시처럼 보임"
```

---

## 2-3. 컨텍스트 스위칭 비용

### 실험: 스레드 수 vs 성능

**환경:** 4코어 CPU, CPU-Bound 작업 (계산)

```c
// 실험 코드
void cpu_bound_task() {
    int sum = 0;
    for (int i = 0; i < 100000000; i++) {
        sum += i;
    }
}

// 결과
스레드 1개:   50ms
스레드 2개:   30ms  (1.6배 빨라짐)
스레드 4개:   20ms  (2.5배 빨라짐) ← 최적!
스레드 8개:   25ms  (2.0배) ← 느려지기 시작
스레드 16개:  35ms  (1.4배) ← 더 느려짐
스레드 32개:  45ms  (1.1배) ← 거의 차이 없음
```

**왜 8개부터 느려지나?**

```
4코어 CPU에 8개 스레드:
  → 컨텍스트 스위칭 발생!
  → SAVE-LOAD-JUMP 반복
  → 실제 계산 시간 < 전환 시간
```

**비용 분석:**
```
컨텍스트 스위칭 1회: 100μs
타이머 인터럽트: 10ms마다

10ms 동안:
- 계산 시간: 9.9ms
- 전환 시간: 0.1ms

스레드가 많으면:
- 계산 시간: 5ms
- 전환 시간: 5ms ← 절반이 낭비!
```

---

## 2-4. ulimit - OS의 안전장치

```bash
# Linux에서 확인
ulimit -u  # 프로세스당 최대 스레드 수

# 출력 예시
4096

# 설정 변경
ulimit -u 2048
```

**왜 제한하나?**
1. 메모리 폭발 방지 (4096 × 1MB = 4GB)
2. Fork Bomb 공격 방어
3. 시스템 안정성

---

# 3. 스레드/프로세스 생명주기와 중요성

## 3-1. 스레드 생명주기 (Thread State Diagram)

```
           create_thread()
                ↓
        ┌───────────────┐
        │      NEW      │  (생성됨, 아직 실행 안 함)
        └───────────────┘
                ↓ start()
        ┌───────────────┐
    ┌──→│    READY      │←──┐  (실행 대기, Ready Queue)
    │   └───────────────┘   │
    │           ↓            │
    │   스케줄러 선택         │  타임 슬라이스 끝
    │           ↓            │  또는 yield()
    │   ┌───────────────┐   │
    │   │   RUNNING     │───┘  (CPU에서 실행 중)
    │   └───────────────┘
    │           ↓
    │   ┌─────────────────────────┐
    │   │ I/O 요청? Lock 대기?    │
    │   └─────────────────────────┘
    │           ↓ Yes
    │   ┌───────────────┐
    └───│   BLOCKED     │  (Wait Queue, CPU 소모 없음!)
        └───────────────┘
                ↓ 이벤트 발생 (I/O 완료, Lock 해제)
        ┌───────────────┐
        │  TERMINATED   │  (종료)
        └───────────────┘
```

---

## 3-2. 각 상태의 의미와 중요성

### READY (준비)

```c
TCB* thread;
thread->state = READY;
enqueue_ready(thread);  // Ready Queue에 대기
```

**의미:**
- "CPU만 주면 바로 실행 가능"
- Ready Queue에서 차례 대기

**중요성:**
- Ready 상태가 너무 많으면? → CPU 경쟁 심함
- 컨텍스트 스위칭 빈번

---

### RUNNING (실행)

```c
thread->state = RUNNING;
current_thread = thread;
// CPU가 thread의 명령어 실행 중
```

**의미:**
- CPU 점유 중
- 실제 작업 수행 중

**중요성:**
- 코어 수만큼만 동시 RUNNING 가능
- 8코어 = 최대 8개 RUNNING

---

### BLOCKED (차단) ⭐ 성능 진단의 핵심!

```c
// DB 락 요청 → 점유됨
thread->state = BLOCKED;
thread->waiting_on = lock_object;
enqueue_wait(thread);  // Wait Queue로 이동
```

**의미:**
- "기다리는 중, CPU 필요 없음"
- Wait Queue에서 대기
- **Ready Queue에 없음 = 스케줄링 대상 아님!**

**중요성 - 성능 진단:**

**스레드 덤프 분석:**
```
"http-nio-8080-exec-1" - BLOCKED
  at CommentService.toggleLike(CommentService.java:42)
  - waiting to lock <0x000000076ab3d3e8>

"http-nio-8080-exec-2" - BLOCKED
  at CommentService.toggleLike(CommentService.java:42)
  - waiting to lock <0x000000076ab3d3e8>

... (50개 스레드 모두 BLOCKED)
```

**진단:**
```
50개 스레드가 BLOCKED
→ 모두 같은 락 기다림
→ 락 경합(Lock Contention)!
→ CommentService:42가 병목!
```

**해결:**
1. 트랜잭션 범위 최소화 (락 점유 시간 단축)
2. 락 방식 변경 (낙관적 락 고려)
3. 쿼리 최적화 (락 대상 최소화)

---

## 3-3. 프로세스 생명주기

```
           fork()
             ↓
    ┌─────────────────┐
    │      NEW        │
    └─────────────────┘
             ↓ exec()
    ┌─────────────────┐
    │    RUNNING      │  (최소 1개 스레드 실행 중)
    └─────────────────┘
             ↓
    ┌─────────────────┐
    │   TERMINATED    │  (모든 스레드 종료)
    └─────────────────┘
             ↓
    ┌─────────────────┐
    │     ZOMBIE      │  (부모가 wait() 안 함)
    └─────────────────┘
```

**Zombie 상태:**
```c
// 프로세스는 종료됐지만 TCB는 메모리에 남음
// 부모 프로세스가 exit code를 읽어가기를 기다림

// 문제: 부모가 wait()를 안 부르면?
// → Zombie 누적 → 메모리 낭비!
```

---

# 4. DB는 비관적 락을 어떻게 구현하는가?

## 4-1. "내가 DB 엔진(InnoDB) 개발자라면?"

### 핵심 자료구조: Lock Manager

```c
// DB 서버 메모리에 존재하는 전역 락 관리자
struct lock_manager {
    hash_table* lock_table;  // Key: 락 대상, Value: Lock Info
    pthread_mutex_t manager_mutex;  // Lock Manager 자체의 동기화
};

// Hash Table의 Key
struct lock_key {
    table_id table_id;
    page_number page_no;
    record_id rec_id;
};

// Hash Table의 Value
struct lock_info {
    // 현재 락 소유자
    transaction_id owner;
    lock_type type;  // SHARED (S) or EXCLUSIVE (X)

    // 대기 큐 (링크드 리스트)
    struct lock_waiter* wait_queue_head;

    // 링크드 리스트
    struct lock_info* next;
};

// 대기 중인 트랜잭션
struct lock_waiter {
    transaction_id tx_id;
    lock_type requested_type;
    struct lock_waiter* next;
};
```

---

## 4-2. SELECT ... FOR UPDATE 내부 동작

### Phase 1: 락 요청 (Lock Request)

```c
// TX_123이 comment_id=5에 대해 FOR UPDATE 실행
int acquire_lock(transaction_id tx_id, lock_key key, lock_type type) {
    // 1. Lock Manager의 Hash Table 조회
    pthread_mutex_lock(&lock_manager.manager_mutex);
    lock_info* lock = hash_table_get(lock_manager.lock_table, key);

    if (lock == NULL) {
        // CASE A: 락 없음 → 즉시 획득 가능
        lock_info* new_lock = (lock_info*)malloc(sizeof(lock_info));
        new_lock->owner = tx_id;
        new_lock->type = type;  // EXCLUSIVE
        new_lock->wait_queue_head = NULL;

        hash_table_put(lock_manager.lock_table, key, new_lock);
        pthread_mutex_unlock(&lock_manager.manager_mutex);

        return LOCK_ACQUIRED;  // 즉시 반환
    }

    // CASE B: 락 존재 → 대기 필요
    if (lock->owner != tx_id) {
        // 대기 큐에 추가
        lock_waiter* waiter = (lock_waiter*)malloc(sizeof(lock_waiter));
        waiter->tx_id = tx_id;
        waiter->requested_type = type;
        waiter->next = NULL;

        // 큐 맨 뒤에 추가
        if (lock->wait_queue_head == NULL) {
            lock->wait_queue_head = waiter;
        } else {
            lock_waiter* tail = lock->wait_queue_head;
            while (tail->next != NULL) {
                tail = tail->next;
            }
            tail->next = waiter;
        }

        pthread_mutex_unlock(&lock_manager.manager_mutex);

        // 트랜잭션 휴면 (DB 세션 sleep)
        // OS 스레드는 Blocked 상태로 전환
        wait_for_lock_release(tx_id, key);

        return LOCK_ACQUIRED;  // 깨어난 후 반환
    }

    pthread_mutex_unlock(&lock_manager.manager_mutex);
    return LOCK_ALREADY_OWNED;
}
```

---

### Phase 2: 트랜잭션 대기

```c
// DB 세션이 여기서 멈춤!
void wait_for_lock_release(transaction_id tx_id, lock_key key) {
    // Condition Variable 사용
    pthread_mutex_lock(&tx_id->mutex);

    while (!is_lock_available(tx_id, key)) {
        // pthread_cond_wait()는:
        // 1. mutex를 unlock하고
        // 2. 조건 변수 신호를 기다림 (sleep)
        // 3. 신호 받으면 깨어나서 mutex를 다시 lock
        pthread_cond_wait(&tx_id->cond_var, &tx_id->mutex);
    }

    pthread_mutex_unlock(&tx_id->mutex);

    // 이제 락 획득 가능!
}
```

**OS 관점:**
```
DB 세션 스레드 (OS Thread):
  state = RUNNING
  → pthread_cond_wait() 호출
  → state = BLOCKED ← 여기서 멈춤!
  → CPU 소모 없음!
```

---

### Phase 3: 락 해제 & 다음 트랜잭션 깨우기

```c
// TX_100이 COMMIT 또는 ROLLBACK 실행
void release_locks(transaction_id tx_id) {
    pthread_mutex_lock(&lock_manager.manager_mutex);

    // 이 트랜잭션이 소유한 모든 락 찾기
    hash_table_iterator* iter = hash_table_iterator_create(lock_manager.lock_table);

    while (hash_table_iterator_has_next(iter)) {
        lock_key key;
        lock_info* lock = hash_table_iterator_next(iter, &key);

        if (lock->owner == tx_id) {
            // 대기 큐 확인
            if (lock->wait_queue_head != NULL) {
                // 맨 앞 트랜잭션을 새 소유자로
                lock_waiter* next_waiter = lock->wait_queue_head;
                lock->wait_queue_head = next_waiter->next;

                lock->owner = next_waiter->tx_id;
                lock->type = next_waiter->requested_type;

                // 대기 중인 트랜잭션 깨우기!
                wakeup_transaction(next_waiter->tx_id);

                free(next_waiter);
            } else {
                // 대기 중인 트랜잭션 없음 → 락 제거
                hash_table_remove(lock_manager.lock_table, key);
                free(lock);
            }
        }
    }

    pthread_mutex_unlock(&lock_manager.manager_mutex);
}

void wakeup_transaction(transaction_id tx_id) {
    pthread_mutex_lock(&tx_id->mutex);
    pthread_cond_signal(&tx_id->cond_var);  // 깨워!
    pthread_mutex_unlock(&tx_id->mutex);
}
```

**OS 관점:**
```
pthread_cond_signal() 호출
  ↓
OS 커널: "아, TX_123을 깨워야겠구나"
  ↓
TX_123 스레드:
  state = BLOCKED → READY
  Ready Queue에 추가
  ↓
스케줄러가 선택
  state = READY → RUNNING
  ↓
wait_for_lock_release() 함수에서 깨어남
  ↓
acquire_lock() 함수 반환
  ↓
SELECT ... FOR UPDATE 결과 반환
```

---

## 4-3. 실제 시나리오: "좋아요" 1000개 요청

### 타임라인

```
t=0ms:
  TX_1~TX_1000: SELECT ... FOR UPDATE 동시 실행

t=0.1ms:
  TX_1: acquire_lock() → 성공! (락 획득)
  TX_2~TX_1000: acquire_lock() → 대기 큐 추가 → sleep()

Lock Manager Hash Table:
  'comment:5' → {
    owner: TX_1,
    type: EXCLUSIVE,
    wait_queue: [TX_2, TX_3, ..., TX_1000]
  }

OS 관점:
  Thread 1: RUNNING (DB 작업 중)
  Thread 2~1000: BLOCKED (락 대기)

t=10ms:
  TX_1: UPDATE ... (count++)
  TX_1: COMMIT
  TX_1: release_locks() 호출
    → Lock Manager: TX_2를 새 owner로
    → wakeup_transaction(TX_2)

OS 관점:
  Thread 1: TERMINATED
  Thread 2: BLOCKED → READY → RUNNING
  Thread 3~1000: BLOCKED

t=20ms:
  TX_2: UPDATE ... (count++)
  TX_2: COMMIT
  TX_2: release_locks()
    → TX_3 깨우기

... (1000번 반복)

t=10000ms (10초):
  모든 트랜잭션 완료
  count = 1000 ✅
```

---

## 4-4. 데드락 감지 (Bonus)

```c
// Lock Manager는 주기적으로 실행
void detect_deadlock(void) {
    // Wait-For Graph 구성
    // TX_A → TX_B: "TX_A가 TX_B가 보유한 락을 기다림"

    graph* wait_for_graph = build_wait_for_graph();

    // 사이클 감지 (DFS)
    if (has_cycle(wait_for_graph)) {
        // 데드락 발견!
        transaction_id victim = choose_victim();  // 가장 적은 작업한 TX 선택

        // Victim 트랜잭션 롤백
        rollback_transaction(victim);
        release_locks(victim);

        // 클라이언트에게 에러 반환
        send_error_to_client(victim, "Deadlock detected");
    }
}
```

---

# 5. 통합: "좋아요" 요청의 전체 여정

## 5-1. 계층별 동작

```
┌─────────────────────────────────────────────────────┐
│ Application Layer (Java)                            │
│ service.toggleLike(commentId, userId);              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ JPA Layer                                           │
│ @Lock(LockModeType.PESSIMISTIC_WRITE)               │
│ → SQL: SELECT ... FOR UPDATE                        │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ JDBC Driver (Thread Pool)                           │
│ OS Thread와 1:1 매핑                                │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ OS Kernel (Linux)                                   │
│ TCB 관리, 스케줄링, 컨텍스트 스위칭                 │
│ BLOCKED 상태 처리                                   │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ DB Engine (MySQL InnoDB)                            │
│ Lock Manager: Hash Table + Wait Queue               │
│ acquire_lock(), release_locks()                     │
└─────────────────────────────────────────────────────┘
```

---

## 5-2. 메모리 사용량 분석

**1000개 요청 처리 시:**

```
스레드 풀: 200개 (고정)
  Stack: 200 × 1MB = 200MB
  TCB: 200 × 9KB = 1.8MB

DB 연결 풀: 200개
  커넥션: 200 × 200KB = 40MB

DB Lock Manager:
  Lock Info: 1개 × 100 bytes = 100 bytes
  Wait Queue: 200개 × 50 bytes = 10KB

총: 약 242MB
```

---

## 5-3. 시간 분석

```
락 획득 대기: 999개 × 10ms = 9990ms (10초)
  → 순차 처리의 대가!

만약 낙관적 락이었다면?
  → 충돌 시 재시도: 약 2초
  → 5배 빠름!

만약 Redis였다면?
  → 메모리 연산: 약 100ms
  → 100배 빠름!
```

---

# 6. 결론: 왜 이런 깊은 이해가 중요한가?

## 6-1. 성능 튜닝

**Before:**
"서버가 느려요" → "스레드 늘려볼까요?" (추측)

**After:**
```
1. 스레드 덤프 → 50개 BLOCKED 발견
2. 모두 CommentService:42에서 대기
3. Lock Manager 분석 → 락 경합 확인
4. 진단: "락 점유 시간이 너무 길다"
5. 해결: 트랜잭션 범위 최소화
6. 결과: 처리 시간 10초 → 1초 (10배 향상)
```

---

## 6-2. 용량 계획

**Before:**
"서버 몇 대 필요할까요?" (감)

**After:**
```
동시 사용자: 10,000명
응답 시간 목표: 100ms
DB 쿼리 시간: 50ms

필요 스레드 수 = 10000 × (100ms / 50ms) = 20,000개?
→ 불가능! (메모리 20GB)

해결:
1. 비동기 처리 → 스레드 100개로 충분
2. Redis 캐싱 → DB 쿼리 시간 5ms
3. 필요 서버: 2대 (각 100 스레드)
```

---

## 6-3. 면접 대응

**질문:** "좋아요에 비관적 락은 과한 거 아닌가요?"

**답변:**
```
"네, 맞습니다.

비관적 락은 DB Lock Manager가 Hash Table과 Wait Queue로
순차 처리를 강제하는 방식입니다.

1000개 요청이면 999개가 Blocked 상태로 대기하고,
OS 스케줄러도 이들을 Ready Queue에서 제외시켜
CPU 자원은 절약되지만, 처리 시간은 10배 느립니다.

좋아요는 약간의 정합성 오차가 허용되므로,
낙관적 락(Version 체크) 또는 Redis 카운터가
더 적합합니다.

이 경험을 통해 '모든 문제에 하나의 정답은 없다'는
것을 배웠습니다."
```

---

**끝.**
