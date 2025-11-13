# 성능 튜닝 종합 가이드

> 실무 중심의 성능 최적화 체크리스트 및 실전 가이드

## 목차
1. [WAS 튜닝](#1-was-튜닝)
2. [DB/쿼리 튜닝](#2-db--쿼리-튜닝)
3. [동기 → 비동기 전환](#3-동기--비동기-전환)
4. [로드 밸런싱](#4-로드-밸런싱)
5. [캐시 최적화](#5-캐시-최적화)
6. [네트워크 튜닝](#6-네트워크-튜닝)
7. [JVM 튜닝](#7-jvm-튜닝)
8. [애플리케이션 레벨 최적화](#8-애플리케이션-레벨-최적화)
9. [파일시스템/디스크 I/O](#9-파일시스템디스크-io)
10. [오브젝트 스토리지](#10-오브젝트-스토리지)
11. [프론트엔드 최적화](#11-프론트엔드-최적화)
12. [클라우드 리소스 최적화](#12-클라우드-리소스-최적화)
13. [CI/CD 최적화](#13-cicd-최적화)
14. [모니터링 및 APM](#14-모니터링-및-apm)

---

## 1. WAS 튜닝

### 1.1 Tomcat 스레드 튜닝

#### 기본 개념
- Tomcat은 **스레드 풀**을 사용해 HTTP 요청을 처리
- 각 요청 = 1개의 스레드가 담당
- 스레드 부족 → 요청 대기 큐에 쌓임 → 응답 지연

#### 주요 설정 파라미터

```xml
<!-- server.xml -->
<Connector port="8080" protocol="HTTP/1.1"
           maxThreads="200"          <!-- 최대 스레드 수 -->
           minSpareThreads="25"      <!-- 최소 유지 스레드 -->
           maxConnections="10000"    <!-- 최대 연결 수 -->
           acceptCount="100"         <!-- 대기 큐 크기 -->
           connectionTimeout="20000" <!-- 연결 타임아웃(ms) -->
           keepAliveTimeout="60000"  <!-- Keep-Alive 타임아웃 -->
           maxKeepAliveRequests="100" />
```

#### Spring Boot 설정

```yaml
# application.yml
server:
  tomcat:
    threads:
      max: 200        # maxThreads
      min-spare: 25   # minSpareThreads
    max-connections: 10000
    accept-count: 100
    connection-timeout: 20s
    keep-alive-timeout: 60s
```

#### 튜닝 전략

**1) 스레드 수 계산 공식**

```
I/O 작업이 많은 경우 (일반적인 웹 애플리케이션):
maxThreads = (CPU 코어 수) × (1 + 대기시간/처리시간)

예시:
- CPU 4 코어
- 평균 요청 처리 시간: 100ms
- 그 중 DB/외부 API 대기: 80ms
- maxThreads = 4 × (1 + 80/20) = 4 × 5 = 20

CPU 집약적인 경우:
maxThreads = CPU 코어 수 + 1
```

**2) Before & After**

```
Before (기본값):
- maxThreads: 200
- 평균 응답 시간: 500ms
- TPS: 150

튜닝 시나리오 1 (I/O 최적화):
- DB 커넥션 풀 증가: 10 → 50
- maxThreads: 200 → 100
- 평균 응답 시간: 500ms → 200ms
- TPS: 150 → 400

이유: 스레드 수보다 DB 커넥션이 병목이었음
```

**3) 모니터링 포인트**

```bash
# 현재 활성 스레드 확인 (JMX)
jconsole 또는 VisualVM 사용

# 스레드 덤프
jstack <pid> > thread_dump.txt

# 주요 지표:
- currentThreadCount: 현재 스레드 수
- currentThreadsBusy: 처리 중인 스레드 수
- 비율이 80% 이상이면 증설 고려
```

### 1.2 Tomcat + OS TCP 튜닝

#### OS 레벨 TCP 설정 (Linux)

```bash
# /etc/sysctl.conf

# 1. TCP 백로그 큐 증가
net.core.somaxconn = 65535                    # Listen 백로그 최대값
net.ipv4.tcp_max_syn_backlog = 8192          # SYN 백로그 큐

# 2. TIME_WAIT 소켓 재사용
net.ipv4.tcp_tw_reuse = 1                    # TIME_WAIT 소켓 재사용
net.ipv4.tcp_fin_timeout = 30                # FIN_WAIT 타임아웃 (기본 60초)

# 3. 포트 범위 확장
net.ipv4.ip_local_port_range = 10000 65535  # 로컬 포트 범위

# 4. TCP 버퍼 크기 조정
net.core.rmem_max = 16777216                 # 수신 버퍼 최대값 (16MB)
net.core.wmem_max = 16777216                 # 송신 버퍼 최대값 (16MB)
net.ipv4.tcp_rmem = 4096 87380 16777216     # TCP 수신 버퍼
net.ipv4.tcp_wmem = 4096 65536 16777216     # TCP 송신 버퍼

# 5. TCP Keep-Alive 설정
net.ipv4.tcp_keepalive_time = 600            # Keep-Alive 시작 시간 (초)
net.ipv4.tcp_keepalive_intvl = 60            # Keep-Alive 재전송 간격
net.ipv4.tcp_keepalive_probes = 3            # Keep-Alive 재전송 횟수

# 적용
sudo sysctl -p
```

#### 파일 디스크립터 제한 증가

```bash
# /etc/security/limits.conf
tomcat soft nofile 65535
tomcat hard nofile 65535

# 현재 제한 확인
ulimit -n

# 프로세스별 확인
cat /proc/<pid>/limits
```

#### 실전 시나리오

```
상황: 트래픽 급증 시 "Too many open files" 에러

원인:
- 기본 ulimit: 1024
- 동시 접속: 5000개
- 각 연결당 소켓 파일 디스크립터 1개

해결:
1. ulimit 증가: 1024 → 65535
2. somaxconn 증가: 128 → 65535
3. Tomcat maxConnections: 10000 → 50000

결과:
- 에러 해소
- 동시 처리 가능 연결: 1024 → 50000
```

---

## 2. DB / 쿼리 튜닝

### 2.1 슬로우 쿼리 튜닝

#### MySQL 슬로우 쿼리 로그 활성화

```sql
-- my.cnf 설정
[mysqld]
slow_query_log = 1
slow_query_log_file = /var/log/mysql/slow-query.log
long_query_time = 1  -- 1초 이상 걸리는 쿼리 기록
log_queries_not_using_indexes = 1

-- 런타임 활성화
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1;
```

#### 쿼리 분석 도구

```bash
# pt-query-digest (Percona Toolkit)
pt-query-digest /var/log/mysql/slow-query.log > report.txt

# 출력 예시:
# Query 1: 1000 QPS, ID 0x1234...
# SELECT * FROM posts WHERE user_id = 123 ORDER BY created_at DESC LIMIT 10
# Execution time: 2.5s
```

#### 실전 튜닝 사례

**Before: N+1 쿼리 문제**

```java
// 게시글 목록 조회 (10개)
List<Post> posts = postRepository.findAll(PageRequest.of(0, 10));

// 각 게시글의 작성자 조회 (10번 추가 쿼리!)
for (Post post : posts) {
    User user = post.getUser();  // Lazy Loading → 10번 SELECT
    System.out.println(user.getName());
}

// 총 쿼리: 1 (게시글) + 10 (작성자) = 11번
// 실행 시간: 550ms
```

**After: Fetch Join**

```java
@Query("SELECT p FROM Post p JOIN FETCH p.user")
List<Post> findAllWithUser(Pageable pageable);

List<Post> posts = postRepository.findAllWithUser(PageRequest.of(0, 10));
for (Post post : posts) {
    System.out.println(post.getUser().getName());  // 추가 쿼리 없음
}

// 총 쿼리: 1번
// 실행 시간: 50ms (11배 개선)
```

### 2.2 인덱스 최적화

#### 인덱스 기본 원리

```
인덱스 = B-Tree 구조 (대부분의 RDBMS)

예시 테이블:
users (id, email, name, created_at)
100만 건 데이터

인덱스 없이 검색:
SELECT * FROM users WHERE email = 'user@example.com';
→ Full Table Scan: 100만 건 전부 확인
→ 실행 시간: 2000ms

인덱스 있으면:
CREATE INDEX idx_email ON users(email);
→ B-Tree 탐색: log2(1,000,000) ≈ 20번 비교
→ 실행 시간: 5ms (400배 빠름)
```

#### 인덱스 생성 전략

**1) 카디널리티 높은 컬럼 우선**

```sql
-- 나쁜 예: 카디널리티 낮음
CREATE INDEX idx_gender ON users(gender);  -- 값이 'M', 'F' 2개뿐
-- 인덱스 효과: 거의 없음 (50% 필터링)

-- 좋은 예: 카디널리티 높음
CREATE INDEX idx_email ON users(email);    -- 값이 거의 유니크
-- 인덱스 효과: 매우 큼 (0.0001% 필터링)
```

**2) 복합 인덱스 순서**

```sql
-- 쿼리 패턴
SELECT * FROM orders
WHERE user_id = 123 AND status = 'PAID' AND created_at > '2025-01-01';

-- 나쁜 예: 순서 잘못됨
CREATE INDEX idx_bad ON orders(created_at, status, user_id);
-- user_id로 먼저 필터링하는데 인덱스 첫 컬럼이 created_at

-- 좋은 예: 등호(=) → 부등호(>) 순서
CREATE INDEX idx_good ON orders(user_id, status, created_at);
-- 1) user_id로 좁히고 → 2) status로 더 좁히고 → 3) created_at으로 정렬

EXPLAIN SELECT * FROM orders
WHERE user_id = 123 AND status = 'PAID' AND created_at > '2025-01-01';

-- Before (인덱스 없음): rows = 1,000,000
-- After (idx_good):     rows = 50  (20,000배 개선)
```

**3) 커버링 인덱스**

```sql
-- 쿼리: 사용자 이메일과 이름만 필요
SELECT email, name FROM users WHERE status = 'ACTIVE';

-- 일반 인덱스
CREATE INDEX idx_status ON users(status);
-- 1) 인덱스에서 status='ACTIVE'인 행 찾기
-- 2) 테이블 가서 email, name 읽기 (Disk I/O 발생)

-- 커버링 인덱스: 필요한 컬럼 전부 포함
CREATE INDEX idx_covering ON users(status, email, name);
-- 1) 인덱스에서 status='ACTIVE'인 행 찾기
-- 2) 인덱스에 email, name도 있음 → 테이블 안 감 (I/O 절약)

-- Before: 10,000 rows × 2번 I/O = 20,000 I/O
-- After:  10,000 rows × 1번 I/O = 10,000 I/O (50% 감소)
```

#### 인덱스 성능 확인

```sql
-- MySQL EXPLAIN 분석
EXPLAIN SELECT * FROM posts WHERE user_id = 123;

주요 필드:
- type:
  - ALL (최악): Full Table Scan
  - index: 인덱스 Full Scan
  - range: 인덱스 범위 스캔
  - ref: 인덱스 특정 값 검색
  - const (최상): Primary Key / Unique 검색

- rows: 예상 검사 행 수 (낮을수록 좋음)
- Extra:
  - Using filesort: 파일 정렬 (느림)
  - Using temporary: 임시 테이블 사용 (느림)
  - Using index: 커버링 인덱스 (빠름)
```

### 2.3 커넥션 풀 튜닝

#### HikariCP 설정 (Spring Boot 기본)

```yaml
# application.yml
spring:
  datasource:
    hikari:
      maximum-pool-size: 10      # 최대 커넥션 수
      minimum-idle: 5            # 최소 유휴 커넥션
      connection-timeout: 30000  # 커넥션 대기 타임아웃 (ms)
      idle-timeout: 600000       # 유휴 커넥션 제거 시간 (10분)
      max-lifetime: 1800000      # 커넥션 최대 수명 (30분)
      leak-detection-threshold: 60000  # 커넥션 누수 감지 (1분)
```

#### 적정 풀 사이즈 계산

```
공식 (HikariCP 권장):
pool_size = (코어 수 × 2) + effective_spindle_count

예시:
- CPU 4코어
- HDD 1개 (effective_spindle_count = 1)
- 권장 풀 사이즈 = 4 × 2 + 1 = 9 → 10

중요: 많다고 좋은 게 아님!
- 풀 사이즈 1000 설정 → DB 서버 부하로 전체 다운
- DB 서버 max_connections도 함께 고려
```

**실전 예시**

```
Before:
- WAS 인스턴스: 3대
- 각 WAS 풀 사이즈: 100
- DB max_connections: 150
- 문제: 3 × 100 = 300 > 150 → DB 연결 거부

After:
- 각 WAS 풀 사이즈: 50
- DB max_connections: 200
- 3 × 50 = 150 < 200 (여유 있음)
- 응답 시간 개선: 풀 대기 시간 감소
```

---

## 3. 동기 → 비동기 전환

### 3.1 이벤트 드리븐 아키텍처

#### 동기 처리의 문제점

```java
// 동기 처리: 주문 생성 API
@PostMapping("/orders")
public Order createOrder(@RequestBody OrderRequest request) {
    Order order = orderService.save(request);        // 100ms
    paymentService.process(order);                   // 500ms (PG사 API)
    emailService.sendConfirmation(order);            // 300ms (SMTP)
    smsService.sendNotification(order);              // 200ms (SMS API)
    inventoryService.decreaseStock(order);           // 150ms

    return order;  // 총 응답 시간: 1250ms
}

문제점:
1. 사용자는 1.25초 대기
2. 외부 API 실패 시 전체 실패
3. 트래픽 증가 시 스레드 고갈
```

#### 비동기 전환 (이벤트 발행)

```java
// 비동기 처리: 이벤트 발행
@PostMapping("/orders")
public Order createOrder(@RequestBody OrderRequest request) {
    Order order = orderService.save(request);        // 100ms

    // 이벤트 발행 (비동기)
    eventPublisher.publish(new OrderCreatedEvent(order));

    return order;  // 응답 시간: 100ms (12.5배 빠름)
}

// 이벤트 핸들러 (별도 스레드/워커)
@EventListener
@Async
public void handleOrderCreated(OrderCreatedEvent event) {
    paymentService.process(event.getOrder());
    emailService.sendConfirmation(event.getOrder());
    smsService.sendNotification(event.getOrder());
    inventoryService.decreaseStock(event.getOrder());
}
```

#### Spring @Async 설정

```java
@Configuration
@EnableAsync
public class AsyncConfig {

    @Bean(name = "taskExecutor")
    public Executor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(10);           // 기본 스레드 수
        executor.setMaxPoolSize(50);            // 최대 스레드 수
        executor.setQueueCapacity(100);         // 큐 사이즈
        executor.setThreadNamePrefix("async-");
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
        executor.initialize();
        return executor;
    }
}
```

### 3.2 메시지 큐 도입

#### Kafka를 활용한 비동기 처리

```java
// Producer: 주문 생성
@Service
public class OrderService {

    @Autowired
    private KafkaTemplate<String, OrderEvent> kafkaTemplate;

    public Order createOrder(OrderRequest request) {
        Order order = orderRepository.save(new Order(request));

        // Kafka로 이벤트 발행
        OrderEvent event = new OrderEvent(order.getId(), order.getUserId());
        kafkaTemplate.send("order-events", event);

        return order;  // 즉시 반환
    }
}

// Consumer: 후속 처리
@Service
public class OrderEventConsumer {

    @KafkaListener(topics = "order-events", groupId = "payment-service")
    public void handlePayment(OrderEvent event) {
        paymentService.process(event.getOrderId());
    }

    @KafkaListener(topics = "order-events", groupId = "notification-service")
    public void handleNotification(OrderEvent event) {
        emailService.send(event.getUserId());
        smsService.send(event.getUserId());
    }
}
```

#### 장점 vs 단점

```
장점:
✓ 응답 시간 단축: 1250ms → 100ms
✓ 서비스 분리: Payment, Notification 독립 배포
✓ 부하 분산: Consumer 스케일 아웃
✓ 재시도: 실패 시 메시지 재처리

단점:
✗ 복잡도 증가: Kafka 인프라 관리
✗ 디버깅 어려움: 분산 추적 필요
✗ 데이터 일관성: Eventually Consistent
✗ 순서 보장: 파티션 전략 필요
```

---

## 4. 로드 밸런싱

### 4.1 L4 vs L7 로드 밸런서

#### L4 (Transport Layer)

```
동작: TCP/UDP 포트 기반 분산

장점:
- 빠름 (패킷 레벨)
- 프로토콜 무관
- 하드웨어 가속 가능

단점:
- 세션 인지 불가
- URL 라우팅 불가
- SSL 터미네이션 제한

예시:
Client → L4 (80 포트) → WAS1, WAS2, WAS3 (라운드로빈)
```

#### L7 (Application Layer)

```
동작: HTTP 헤더/URL/쿠키 기반 분산

장점:
- URL 기반 라우팅
  - /api/* → API 서버
  - /static/* → Static 서버
- 세션 스티키 (Cookie 기반)
- SSL 오프로딩
- 헬스 체크 (HTTP 200)

단점:
- L4보다 느림 (HTTP 파싱)
- CPU 사용량 높음

예시 (Nginx):
```

```nginx
upstream api_servers {
    server 192.168.1.101:8080;
    server 192.168.1.102:8080;
    server 192.168.1.103:8080;
}

upstream static_servers {
    server 192.168.1.201:80;
    server 192.168.1.202:80;
}

server {
    listen 80;

    # API 요청 → API 서버
    location /api/ {
        proxy_pass http://api_servers;
    }

    # 정적 파일 → Static 서버
    location /static/ {
        proxy_pass http://static_servers;
    }
}
```

### 4.2 로드 밸런싱 알고리즘

#### 1) Round Robin (순환)

```nginx
upstream backend {
    server server1.example.com;
    server server2.example.com;
    server server3.example.com;
}

요청 1 → server1
요청 2 → server2
요청 3 → server3
요청 4 → server1 (반복)

장점: 단순, 균등 분산
단점: 서버 성능 차이 무시
```

#### 2) Least Connections (최소 연결)

```nginx
upstream backend {
    least_conn;
    server server1.example.com;
    server server2.example.com;
    server server3.example.com;
}

현재 상태:
- server1: 10개 연결
- server2: 5개 연결
- server3: 15개 연결

다음 요청 → server2 (연결 수 가장 적음)

장점: 부하 균등화
단점: 연결 수 ≠ 실제 부하
```

#### 3) IP Hash (세션 유지)

```nginx
upstream backend {
    ip_hash;
    server server1.example.com;
    server server2.example.com;
    server server3.example.com;
}

Client IP: 123.45.67.89
hash(123.45.67.89) % 3 = 1 → 항상 server2로

장점: 세션 유지 (Sticky Session)
단점: 특정 서버 집중 가능
```

#### 4) Weighted (가중치)

```nginx
upstream backend {
    server server1.example.com weight=3;  # 고성능 서버
    server server2.example.com weight=2;
    server server3.example.com weight=1;  # 저성능 서버
}

6개 요청:
- server1: 3개 (50%)
- server2: 2개 (33%)
- server3: 1개 (17%)

장점: 서버 성능 반영
단점: 수동 설정 필요
```

### 4.3 헬스 체크 설정

```nginx
upstream backend {
    server backend1.example.com max_fails=3 fail_timeout=30s;
    server backend2.example.com max_fails=3 fail_timeout=30s;
    server backend3.example.com max_fails=3 fail_timeout=30s;
}

# max_fails=3: 3번 실패 시 제외
# fail_timeout=30s: 30초 후 재시도

헬스 체크 엔드포인트:
```

```java
@RestController
public class HealthController {

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        // DB 연결 체크
        if (!dbHealthCheck()) {
            return ResponseEntity.status(503).body("DB connection failed");
        }

        return ResponseEntity.ok("OK");
    }
}
```

---

## 5. 캐시 최적화

### 5.1 캐시 레이어 전략

```
계층별 캐시:

┌─────────────────┐
│   Browser       │ ← 1. 브라우저 캐시 (로컬)
└─────────────────┘
        ↓
┌─────────────────┐
│   CDN           │ ← 2. CDN 캐시 (엣지)
└─────────────────┘
        ↓
┌─────────────────┐
│   Nginx         │ ← 3. 리버스 프록시 캐시
└─────────────────┘
        ↓
┌─────────────────┐
│   Redis         │ ← 4. 애플리케이션 캐시
└─────────────────┘
        ↓
┌─────────────────┐
│   Database      │ ← 5. DB 쿼리 캐시
└─────────────────┘
```

### 5.2 Redis 캐시 전략

#### Cache-Aside (Lazy Loading)

```java
@Service
public class UserService {

    @Autowired
    private RedisTemplate<String, User> redisTemplate;

    @Autowired
    private UserRepository userRepository;

    public User getUser(Long userId) {
        String key = "user:" + userId;

        // 1. 캐시 확인
        User cached = redisTemplate.opsForValue().get(key);
        if (cached != null) {
            return cached;  // 캐시 히트
        }

        // 2. DB 조회
        User user = userRepository.findById(userId)
            .orElseThrow(() -> new UserNotFoundException());

        // 3. 캐시 저장 (TTL 1시간)
        redisTemplate.opsForValue().set(key, user, 1, TimeUnit.HOURS);

        return user;
    }
}

성능 개선:
- 캐시 히트: 1ms (Redis 조회)
- 캐시 미스: 50ms (DB 조회)
- 히트율 90% 가정: 평균 5.9ms (0.9×1 + 0.1×50)
```

#### Write-Through (동기화 쓰기)

```java
public void updateUser(User user) {
    // 1. DB 업데이트
    userRepository.save(user);

    // 2. 캐시 업데이트
    String key = "user:" + user.getId();
    redisTemplate.opsForValue().set(key, user, 1, TimeUnit.HOURS);
}

장점: 캐시-DB 일관성 보장
단점: 쓰기 지연 (2번 작업)
```

#### Write-Behind (비동기 쓰기)

```java
public void updateUser(User user) {
    // 1. 캐시만 업데이트 (빠름)
    String key = "user:" + user.getId();
    redisTemplate.opsForValue().set(key, user, 1, TimeUnit.HOURS);

    // 2. DB 쓰기는 백그라운드로 (큐에 넣기)
    writeQueue.offer(user);
}

장점: 쓰기 속도 매우 빠름
단점: 캐시 장애 시 데이터 손실 위험
```

### 5.3 캐시 Stampede 방지

#### 문제 상황

```
시나리오: 인기 상품 캐시 만료

시간 0초: 캐시 만료
시간 0.1초: 1000개 동시 요청 → 모두 캐시 미스
→ 1000개 요청이 동시에 DB 조회
→ DB 부하로 다운

Time │ Request │ Action
─────┼─────────┼────────────────
0.0s │ 1~1000  │ Cache miss!
0.1s │ 1~1000  │ SELECT * FROM products WHERE id=123
0.2s │ -       │ DB overload! Connection timeout!
```

#### 해결 1: 분산 락

```java
public Product getProduct(Long productId) {
    String cacheKey = "product:" + productId;
    String lockKey = "lock:product:" + productId;

    // 1. 캐시 확인
    Product cached = redisTemplate.opsForValue().get(cacheKey);
    if (cached != null) {
        return cached;
    }

    // 2. 분산 락 획득 시도 (최대 5초 대기)
    Boolean locked = redisTemplate.opsForValue()
        .setIfAbsent(lockKey, "1", 10, TimeUnit.SECONDS);

    if (locked) {
        try {
            // 3. DB 조회 (1개 요청만 실행)
            Product product = productRepository.findById(productId)
                .orElseThrow();

            // 4. 캐시 저장
            redisTemplate.opsForValue().set(cacheKey, product, 1, TimeUnit.HOURS);

            return product;
        } finally {
            // 5. 락 해제
            redisTemplate.delete(lockKey);
        }
    } else {
        // 6. 락 획득 실패 → 다른 스레드가 로딩 중
        // 잠시 대기 후 캐시 재조회
        Thread.sleep(100);
        return getProduct(productId);  // 재시도
    }
}

결과:
- 1000개 요청 중 1개만 DB 조회
- 나머지 999개는 대기 후 캐시에서 조회
```

#### 해결 2: 확률적 조기 갱신

```java
public Product getProduct(Long productId) {
    String key = "product:" + productId;

    // 캐시 조회 (TTL 정보 포함)
    ValueWrapper wrapper = redisTemplate.opsForValue().get(key);

    if (wrapper != null) {
        long ttl = wrapper.getTtl();  // 남은 TTL (초)
        double refreshProbability = Math.log(ttl) / ttl;

        // 확률적으로 조기 갱신 (TTL 짧을수록 확률 높음)
        if (Math.random() < refreshProbability) {
            // 백그라운드에서 갱신
            CompletableFuture.runAsync(() -> refreshCache(productId));
        }

        return wrapper.getValue();
    }

    // 캐시 미스: DB 조회
    return loadFromDB(productId);
}

예시:
- TTL 3600초 (1시간): 갱신 확률 0.2%
- TTL 60초: 갱신 확률 6.8%
- TTL 10초: 갱신 확률 23%
→ 만료 임박 시 미리 갱신
```

### 5.4 CDN 캐시 활용

#### CloudFront 설정 예시

```yaml
# CloudFront Distribution
CacheBehavior:
  PathPattern: /static/*
  TargetOrigin: S3-bucket
  ViewerProtocolPolicy: redirect-to-https
  CachePolicyId: CachingOptimized

  # 캐시 TTL
  MinTTL: 0
  DefaultTTL: 86400      # 1일
  MaxTTL: 31536000       # 1년

  # 캐시 키 설정
  CacheKeyParameters:
    QueryStrings: none    # 쿼리 스트링 무시
    Headers:
      - CloudFront-Viewer-Country  # 국가별 캐시
    Cookies: none
```

#### Cache-Control 헤더 설정

```java
@GetMapping("/static/image.jpg")
public ResponseEntity<byte[]> getImage() {
    byte[] image = loadImage();

    return ResponseEntity.ok()
        .cacheControl(CacheControl.maxAge(30, TimeUnit.DAYS)
            .cachePublic()         // CDN 캐시 허용
            .immutable())          // 변경 불가 (적극 캐싱)
        .body(image);
}

응답 헤더:
Cache-Control: max-age=2592000, public, immutable

→ 브라우저/CDN에서 30일간 캐싱
→ 30일 내 재요청 시 서버 도달 안 함 (Bandwidth 절약)
```

---

## 6. 네트워크 튜닝

### 6.1 OS 레벨 TCP 최적화

#### TCP Window Size 조정

```bash
# /etc/sysctl.conf

# TCP 윈도우 스케일링 (대역폭×지연 제품 최적화)
net.ipv4.tcp_window_scaling = 1

# 수신 윈도우 크기
net.ipv4.tcp_rmem = 4096 87380 16777216
# 최소 4KB, 기본 85KB, 최대 16MB

# 송신 윈도우 크기
net.ipv4.tcp_wmem = 4096 65536 16777216

# BDP (Bandwidth-Delay Product) 계산
대역폭: 100 Mbps = 12.5 MB/s
RTT: 100ms = 0.1s
BDP = 12.5 MB/s × 0.1s = 1.25 MB

최적 윈도우 크기 ≥ 1.25 MB
→ tcp_rmem 최대값 16MB로 충분
```

#### TCP 혼잡 제어 알고리즘

```bash
# 사용 가능한 알고리즘 확인
sysctl net.ipv4.tcp_available_congestion_control

# BBR (Bottleneck Bandwidth and RTT) 설정
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr

BBR vs CUBIC (기본값):
- CUBIC: 패킷 손실 기반 (손실 발생 시 속도 감소)
- BBR: 대역폭 기반 (실제 네트워크 용량 추정)

성능 개선:
- 고지연 네트워크: 2~3배 처리량 증가
- 패킷 손실 환경: 더 안정적인 속도
```

### 6.2 HTTP/2 및 압축

#### Nginx HTTP/2 설정

```nginx
server {
    listen 443 ssl http2;  # HTTP/2 활성화
    server_name example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Gzip 압축
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;       # 1KB 이상만 압축
    gzip_comp_level 6;          # 압축 레벨 (1~9)
    gzip_types
        text/plain
        text/css
        application/json
        application/javascript
        application/xml
        image/svg+xml;

    # Brotli 압축 (Gzip보다 15~25% 더 압축)
    brotli on;
    brotli_comp_level 6;
    brotli_types
        text/plain
        text/css
        application/json
        application/javascript;
}
```

#### HTTP/2 성능 개선

```
HTTP/1.1의 문제:
- Head-of-Line Blocking: 요청 순차 처리
- 연결당 1개 요청: 여러 파일 로드 시 여러 연결 필요

HTTP/2 개선:
- Multiplexing: 1개 연결로 여러 요청 동시 처리
- Server Push: 요청 전에 미리 전송
- Header 압축: HPACK 알고리즘

예시:
HTML + 10개 CSS + 20개 JS + 50개 이미지 = 81개 파일

HTTP/1.1:
- 6개 연결 (브라우저 제한)
- 81 ÷ 6 ≈ 14 라운드트립
- 총 시간: 14 × RTT = 14 × 100ms = 1.4초

HTTP/2:
- 1개 연결
- 81개 파일 동시 전송 (Multiplexing)
- 총 시간: 1 × RTT = 100ms (14배 빠름)
```

### 6.3 Connection Pooling

#### HTTP Client 커넥션 풀

```java
// Apache HttpClient 설정
@Configuration
public class HttpClientConfig {

    @Bean
    public HttpClient httpClient() {
        PoolingHttpClientConnectionManager connectionManager =
            new PoolingHttpClientConnectionManager();

        // 전체 최대 연결 수
        connectionManager.setMaxTotal(200);

        // 호스트당 최대 연결 수
        connectionManager.setDefaultMaxPerRoute(20);

        // 특정 호스트 커스텀 설정
        HttpRoute route = new HttpRoute(new HttpHost("api.example.com", 443, "https"));
        connectionManager.setMaxPerRoute(route, 50);

        // 유휴 연결 정리
        connectionManager.setValidateAfterInactivity(2000);  // 2초

        return HttpClients.custom()
            .setConnectionManager(connectionManager)
            .setConnectionTimeToLive(30, TimeUnit.SECONDS)
            .build();
    }
}

성능 개선:
- 연결 재사용: TCP 핸드셰이크 생략 (RTT 절약)
- Before (연결 생성): 100ms (TCP) + 50ms (SSL) + 10ms (요청) = 160ms
- After (연결 재사용): 10ms (요청만) (16배 빠름)
```

---

## 7. JVM 튜닝

### 7.1 힙 메모리 설정

#### 기본 JVM 옵션

```bash
# 힙 사이즈 설정
java -Xms2g -Xmx2g \          # 초기/최대 힙 (같게 설정 권장)
     -XX:MetaspaceSize=256m \ # Metaspace 초기 크기
     -XX:MaxMetaspaceSize=512m \ # Metaspace 최대 크기
     -jar application.jar

권장 사항:
1. -Xms = -Xmx: 힙 리사이징 오버헤드 제거
2. 전체 RAM의 50~75%: OS/기타 프로세스 고려
3. 32GB 이하: Compressed OOP 활용

예시 (16GB RAM 서버):
-Xms8g -Xmx8g
```

#### Young Generation vs Old Generation

```
힙 구조:
┌───────────────────────────────────┐
│         Young Generation          │ ← 새 객체 생성
│  ┌──────┬──────────┬──────────┐  │
│  │ Eden │ Survivor │ Survivor │  │
│  │      │    0     │    1     │  │
│  └──────┴──────────┴──────────┘  │
├───────────────────────────────────┤
│         Old Generation            │ ← 오래된 객체
│                                   │
└───────────────────────────────────┘

Young/Old 비율 조정:
-XX:NewRatio=2  # Old : Young = 2 : 1
-XX:NewSize=512m -XX:MaxNewSize=512m  # Young 고정 크기
```

### 7.2 가비지 컬렉터 선택

#### G1GC (Java 9+ 기본)

```bash
java -XX:+UseG1GC \
     -XX:MaxGCPauseMillis=200 \    # 목표 정지 시간 (ms)
     -XX:G1HeapRegionSize=16m \    # 리전 크기
     -XX:InitiatingHeapOccupancyPercent=45 \  # GC 시작 임계값
     -Xms8g -Xmx8g \
     -jar application.jar

특징:
- 대용량 힙 (> 4GB)에 적합
- 예측 가능한 정지 시간
- 리전 기반 관리

사용 사례:
- 웹 애플리케이션 (응답 시간 중요)
- 마이크로서비스
```

#### ZGC (Java 15+)

```bash
java -XX:+UseZGC \
     -XX:ZCollectionInterval=120 \  # GC 간격 (초)
     -Xms16g -Xmx16g \
     -jar application.jar

특징:
- 정지 시간 < 10ms (대부분 < 1ms)
- 대용량 힙 (수 TB)까지 확장 가능
- Concurrent GC (애플리케이션과 동시 실행)

사용 사례:
- 초저지연 요구사항 (금융, 게임)
- 대용량 메모리 (수십 GB~TB)
```

#### 성능 비교

```
테스트: 8GB 힙, 웹 애플리케이션

Serial GC (단일 스레드):
- GC 시간: 5초
- 처리량: 낮음
- 사용처: 소형 애플리케이션

Parallel GC (멀티 스레드):
- GC 시간: 1초
- 처리량: 높음 (95%)
- 정지 시간: 길어질 수 있음

G1GC:
- GC 시간: 200ms (목표치)
- 처리량: 중간 (90%)
- 정지 시간: 예측 가능

ZGC:
- GC 시간: 5ms
- 처리량: 높음 (92%)
- 정지 시간: 매우 짧음
```

### 7.3 GC 로그 분석

```bash
# GC 로그 활성화 (Java 11+)
java -Xlog:gc*:file=/var/log/gc.log:time,level,tags \
     -XX:+UseG1GC \
     -jar application.jar

# 로그 예시
[2025-11-13T10:15:30.123+0900][info][gc] GC(123) Pause Young (Normal) 512M->128M(2048M) 45.678ms
[2025-11-13T10:16:45.456+0900][info][gc] GC(124) Pause Full (Allocation Failure) 1800M->600M(2048M) 2345.678ms

분석:
- Pause Young: Minor GC (빈번, 짧음)
- Pause Full: Full GC (드물게, 길면 문제)
- 512M->128M: GC 전후 힙 사용량
- 45.678ms: 정지 시간

경고 신호:
✗ Full GC 빈번 발생: 힙 부족 또는 메모리 누수
✗ 정지 시간 증가 추세: 튜닝 필요
✗ Old Generation 사용률 > 90%: 힙 증설 고려
```

#### GCEasy.io를 활용한 분석

```bash
# GC 로그 수집
curl -X POST -F "file=@gc.log" https://api.gceasy.io/analyzeGC

주요 메트릭:
- Throughput: 99.5% (목표: > 95%)
- Avg Pause Time: 50ms (목표: < 100ms)
- Max Pause Time: 500ms (목표: < 1s)
- Memory Leak 탐지: Old Gen 증가 패턴
```

---

## 8. 애플리케이션 레벨 최적화

### 8.1 리소스 관리

#### Connection Leak 방지

```java
// 나쁜 예: 연결 누수
public void badExample() {
    Connection conn = dataSource.getConnection();
    Statement stmt = conn.createStatement();
    ResultSet rs = stmt.executeQuery("SELECT * FROM users");
    // ... 처리
    // ❌ close() 누락 → 연결 고갈
}

// 좋은 예: try-with-resources
public void goodExample() {
    String sql = "SELECT * FROM users";

    try (Connection conn = dataSource.getConnection();
         Statement stmt = conn.createStatement();
         ResultSet rs = stmt.executeQuery(sql)) {

        while (rs.next()) {
            // ... 처리
        }
    } // ✅ 자동으로 close()
}
```

#### 트랜잭션 범위 최소화

```java
// 나쁜 예: 트랜잭션이 너무 김
@Transactional
public void processOrder(OrderRequest request) {
    Order order = orderRepository.save(new Order(request));  // 100ms

    // ❌ 외부 API 호출을 트랜잭션 안에서!
    PaymentResult payment = externalPaymentAPI.charge(order);  // 500ms

    order.setPaymentId(payment.getId());
    orderRepository.save(order);  // 50ms

    // 트랜잭션 시간: 650ms (DB 커넥션 점유)
}

// 좋은 예: 트랜잭션 분리
public void processOrder(OrderRequest request) {
    // 1. 주문 생성 (트랜잭션 1)
    Order order = createOrder(request);  // 100ms, 트랜잭션 종료

    // 2. 외부 API 호출 (트랜잭션 없음)
    PaymentResult payment = externalPaymentAPI.charge(order);  // 500ms

    // 3. 결제 정보 업데이트 (트랜잭션 2)
    updatePayment(order, payment);  // 50ms, 트랜잭션 종료

    // 최대 트랜잭션 시간: 100ms (6.5배 개선)
}

@Transactional
private Order createOrder(OrderRequest request) {
    return orderRepository.save(new Order(request));
}

@Transactional
private void updatePayment(Order order, PaymentResult payment) {
    order.setPaymentId(payment.getId());
    orderRepository.save(order);
}
```

### 8.2 세션 관리

#### Stateless 아키텍처 전환

```java
// Before: 세션에 사용자 정보 저장
@Controller
public class OldController {

    @PostMapping("/login")
    public String login(@RequestParam String username, HttpSession session) {
        User user = userService.authenticate(username);
        session.setAttribute("user", user);  // 세션에 저장 (메모리 사용)
        return "redirect:/dashboard";
    }

    @GetMapping("/dashboard")
    public String dashboard(HttpSession session, Model model) {
        User user = (User) session.getAttribute("user");  // 세션 조회
        model.addAttribute("user", user);
        return "dashboard";
    }
}

문제점:
- 서버 메모리 사용 (세션 × 사용자 수)
- Sticky Session 필요 (특정 서버에 고정)
- 스케일 아웃 어려움

// After: JWT 토큰 사용 (Stateless)
@RestController
public class NewController {

    @PostMapping("/login")
    public LoginResponse login(@RequestBody LoginRequest request) {
        User user = userService.authenticate(request.getUsername());

        // JWT 생성 (서버 메모리 사용 없음)
        String token = jwtService.generateToken(user);

        return new LoginResponse(token);
    }

    @GetMapping("/dashboard")
    public Dashboard dashboard(@RequestHeader("Authorization") String token) {
        // JWT 검증 및 사용자 정보 추출
        User user = jwtService.parseToken(token);

        return dashboardService.getDashboard(user);
    }
}

장점:
✓ 서버 메모리 절약
✓ 수평 확장 자유로움
✓ 로드 밸런서 단순화 (Sticky 불필요)
```

### 8.3 로깅 최적화

#### 비동기 로깅

```xml
<!-- logback.xml -->
<configuration>
    <!-- 동기 Appender (느림) -->
    <appender name="FILE_SYNC" class="ch.qos.logback.core.FileAppender">
        <file>/var/log/app.log</file>
        <encoder>
            <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n</pattern>
        </encoder>
    </appender>

    <!-- 비동기 Appender (빠름) -->
    <appender name="FILE_ASYNC" class="ch.qos.logback.classic.AsyncAppender">
        <appender-ref ref="FILE_SYNC" />
        <queueSize>512</queueSize>           <!-- 큐 사이즈 -->
        <discardingThreshold>0</discardingThreshold>  <!-- 버리지 않음 -->
        <neverBlock>false</neverBlock>       <!-- 큐 가득 차면 대기 -->
    </appender>

    <root level="INFO">
        <appender-ref ref="FILE_ASYNC" />
    </root>
</configuration>

성능 비교:
- 동기 로깅: 10,000건 쓰기 = 500ms (Disk I/O 대기)
- 비동기 로깅: 10,000건 쓰기 = 5ms (큐에 넣고 반환)
```

#### 로그 레벨 최적화

```java
// 나쁜 예: 문자열 연결 (로그 출력 안 해도 연산 발생)
logger.debug("User info: " + user.getName() + ", " + user.getEmail());
// DEBUG 레벨 꺼져있어도 문자열 연산 실행됨!

// 좋은 예 1: 파라미터 방식
logger.debug("User info: {}, {}", user.getName(), user.getEmail());
// DEBUG 꺼져있으면 getName(), getEmail() 호출 안 함

// 좋은 예 2: 조건 체크
if (logger.isDebugEnabled()) {
    logger.debug("Expensive calculation: {}", expensiveMethod());
}

// 프로덕션 설정
logging.level.root=INFO              # DEBUG 로그 제거
logging.level.org.hibernate.SQL=WARN # SQL 로그 최소화
```

---

## 9. 파일시스템/디스크 I/O

### 9.1 디스크 I/O 모니터링

```bash
# iostat: 디스크 사용량 확인
iostat -x 1

주요 지표:
- %util: 디스크 사용률 (100% = 병목)
- await: 평균 I/O 대기 시간 (ms)
  - HDD: 10~20ms 정상
  - SSD: 1ms 이하 정상
- r/s, w/s: 초당 읽기/쓰기 횟수

예시:
Device   r/s   w/s   await  %util
sda     500   200    15.0    95%   ← 병목 발생!
sdb      50    20     2.0    20%   ← 여유 있음
```

### 9.2 파일 시스템 튜닝

#### noatime 옵션 (읽기 성능 개선)

```bash
# /etc/fstab
/dev/sda1  /data  ext4  defaults,noatime,nodiratime  0  2

# noatime: 파일 접근 시간 기록 안 함
# nodiratime: 디렉토리 접근 시간 기록 안 함

성능 개선:
- 읽기 작업마다 메타데이터 쓰기 발생 → 제거
- I/O 횟수 30~40% 감소
```

#### 임시 파일은 tmpfs 사용

```bash
# tmpfs: RAM 기반 파일시스템 (매우 빠름)
mount -t tmpfs -o size=1G tmpfs /tmp

# /etc/fstab에 추가
tmpfs  /tmp  tmpfs  defaults,size=1G,mode=1777  0  0

속도 비교:
- HDD /tmp: 100 MB/s
- SSD /tmp: 500 MB/s
- tmpfs:    5000 MB/s (50배 빠름)

주의: 재부팅 시 데이터 손실 (임시 파일만 사용)
```

### 9.3 대용량 파일 처리 최적화

#### 스트리밍 처리

```java
// 나쁜 예: 전체 파일을 메모리에 로드
public List<String> processFile(String path) throws IOException {
    List<String> lines = Files.readAllLines(Paths.get(path));
    // ❌ 1GB 파일 → 1GB 메모리 사용 → OutOfMemoryError

    return lines.stream()
        .filter(line -> line.contains("ERROR"))
        .collect(Collectors.toList());
}

// 좋은 예: 스트림으로 한 줄씩 처리
public void processFile(String path) throws IOException {
    try (Stream<String> lines = Files.lines(Paths.get(path))) {
        lines.filter(line -> line.contains("ERROR"))
             .forEach(line -> {
                 // 한 줄씩 처리 (메모리 사용량 일정)
                 processErrorLine(line);
             });
    }
    // ✅ 1GB 파일도 메모리 몇 MB만 사용
}
```

---

## 10. 오브젝트 스토리지

### 10.1 S3 성능 최적화

#### 멀티파트 업로드

```java
@Service
public class S3Service {

    @Autowired
    private AmazonS3 s3Client;

    // 대용량 파일 업로드 (100MB 이상)
    public void uploadLargeFile(String bucket, String key, File file) {
        // 1. 멀티파트 업로드 시작
        InitiateMultipartUploadRequest initRequest =
            new InitiateMultipartUploadRequest(bucket, key);
        InitiateMultipartUploadResult initResult =
            s3Client.initiateMultipartUpload(initRequest);

        String uploadId = initResult.getUploadId();
        long partSize = 10 * 1024 * 1024;  // 10MB 파트

        List<PartETag> partETags = new ArrayList<>();
        long filePosition = 0;

        // 2. 파일을 10MB씩 분할하여 병렬 업로드
        for (int i = 1; filePosition < file.length(); i++) {
            long currentPartSize = Math.min(partSize, file.length() - filePosition);

            UploadPartRequest uploadRequest = new UploadPartRequest()
                .withBucketName(bucket)
                .withKey(key)
                .withUploadId(uploadId)
                .withPartNumber(i)
                .withFileOffset(filePosition)
                .withFile(file)
                .withPartSize(currentPartSize);

            PartETag partETag = s3Client.uploadPart(uploadRequest).getPartETag();
            partETags.add(partETag);

            filePosition += currentPartSize;
        }

        // 3. 업로드 완료
        CompleteMultipartUploadRequest compRequest =
            new CompleteMultipartUploadRequest(bucket, key, uploadId, partETags);
        s3Client.completeMultipartUpload(compRequest);
    }
}

성능 개선:
- 단일 PUT: 1GB 파일 → 60초
- 멀티파트 (10MB × 100파트 병렬): 1GB → 15초 (4배 빠름)
```

#### Transfer Acceleration

```java
// S3 Transfer Acceleration 활성화
AmazonS3 s3Client = AmazonS3ClientBuilder.standard()
    .withAccelerateModeEnabled(true)  // CloudFront 엣지 로케이션 활용
    .build();

속도 비교 (서울 → 미국 버킷):
- 일반 업로드: 5 MB/s
- Transfer Acceleration: 25 MB/s (5배 빠름)

비용: 추가 요금 발생 (GB당 $0.04)
```

### 10.2 CloudFront + S3 조합

```yaml
# CloudFront Distribution 설정
Origin:
  DomainName: my-bucket.s3.amazonaws.com
  OriginAccessIdentity: E1234567890ABC  # S3 직접 접근 차단

Behaviors:
  PathPattern: /images/*
  ViewerProtocolPolicy: https-only
  CachePolicyId: CachingOptimized
  TTL:
    Min: 86400      # 1일
    Default: 604800 # 7일
    Max: 31536000   # 1년

성능 개선:
- 첫 요청: S3에서 로드 (500ms)
- 이후 요청: CloudFront 엣지에서 (50ms, 10배 빠름)
- 대역폭 비용: S3 → 인터넷 ($0.09/GB) → CloudFront ($0.085/GB) 절약
```

---

## 11. 프론트엔드 최적화

### 11.1 번들 크기 최적화

#### Code Splitting (React 예시)

```javascript
// Before: 전체 번들 로드
import Dashboard from './Dashboard';
import Profile from './Profile';
import Settings from './Settings';

function App() {
  return (
    <Router>
      <Route path="/dashboard" component={Dashboard} />
      <Route path="/profile" component={Profile} />
      <Route path="/settings" component={Settings} />
    </Router>
  );
}

// 번들 크기: 2MB (모든 페이지 포함)
// 초기 로드: 2MB 다운로드 → 5초

// After: 동적 import로 Code Splitting
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./Dashboard'));
const Profile = lazy(() => import('./Profile'));
const Settings = lazy(() => import('./Settings'));

function App() {
  return (
    <Router>
      <Suspense fallback={<Loading />}>
        <Route path="/dashboard" component={Dashboard} />
        <Route path="/profile" component={Profile} />
        <Route path="/settings" component={Settings} />
      </Suspense>
    </Router>
  );
}

// 초기 번들: 500KB (공통 코드만)
// 각 페이지: 필요할 때 로드
// 초기 로드: 500KB → 1초 (5배 빠름)
```

#### Tree Shaking

```javascript
// 나쁜 예: 전체 라이브러리 import
import _ from 'lodash';  // 전체 70KB
_.debounce(fn, 300);

// 좋은 예: 필요한 함수만 import
import debounce from 'lodash/debounce';  // 5KB (14배 감소)
debounce(fn, 300);

// Webpack 설정
module.exports = {
  mode: 'production',  // Tree shaking 자동 활성화
  optimization: {
    usedExports: true,  // 사용되지 않는 코드 제거
  },
};
```

### 11.2 이미지 최적화

#### 포맷 선택

```html
<!-- WebP 지원 브라우저용 + JPEG 폴백 -->
<picture>
  <source srcset="image.webp" type="image/webp">
  <source srcset="image.jpg" type="image/jpeg">
  <img src="image.jpg" alt="Product">
</picture>

파일 크기 비교 (동일 품질):
- PNG:  1000 KB
- JPEG: 300 KB (3.3배 감소)
- WebP: 150 KB (6.7배 감소)
```

#### Lazy Loading

```html
<!-- 브라우저 네이티브 Lazy Loading -->
<img src="image.jpg" loading="lazy" alt="Product">

<!-- Intersection Observer API -->
<script>
const images = document.querySelectorAll('img[data-src]');
const imageObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const img = entry.target;
      img.src = img.dataset.src;  // 실제 이미지 로드
      imageObserver.unobserve(img);
    }
  });
});

images.forEach(img => imageObserver.observe(img));
</script>

효과:
- 페이지 로드: 100개 이미지 × 200KB = 20MB → 보이는 5개만 로드 = 1MB
- 초기 로딩 시간: 10초 → 0.5초 (20배 빠름)
```

### 11.3 렌더링 최적화

#### Virtual DOM 최적화 (React)

```javascript
// 나쁜 예: 불필요한 리렌더링
function ProductList({ products }) {
  return (
    <div>
      {products.map(product => (
        <ProductCard key={product.id} product={product} />
      ))}
    </div>
  );
}

function ProductCard({ product }) {
  // 부모가 리렌더링되면 모든 카드 리렌더링
  return <div>{product.name}</div>;
}

// 좋은 예: React.memo로 최적화
const ProductCard = React.memo(({ product }) => {
  return <div>{product.name}</div>;
}, (prevProps, nextProps) => {
  // product가 변경되지 않으면 리렌더링 생략
  return prevProps.product.id === nextProps.product.id;
});

성능 개선:
- Before: 100개 상품, 1개 변경 → 100개 전부 리렌더링
- After: 100개 상품, 1개 변경 → 1개만 리렌더링 (100배 개선)
```

---

## 12. 클라우드 리소스 최적화

### 12.1 인스턴스 타입 최적화

#### CPU vs Memory 집약적 워크로드

```
AWS EC2 인스턴스 타입:

1. 범용 (T3, M5):
   - CPU:Memory = 1:4
   - t3.medium: 2 vCPU, 4GB RAM
   - 사용처: 웹 서버, 소형 DB

2. CPU 최적화 (C5):
   - CPU:Memory = 1:2
   - c5.large: 2 vCPU, 4GB RAM → c5.xlarge: 4 vCPU, 8GB
   - 사용처: 배치 처리, 동영상 인코딩

3. 메모리 최적화 (R5):
   - CPU:Memory = 1:8
   - r5.large: 2 vCPU, 16GB RAM
   - 사용처: Redis, Memcached, DB

4. 스토리지 최적화 (I3):
   - 로컬 NVMe SSD
   - 사용처: NoSQL, 데이터 웨어하우스

비용 최적화 예시:
- Before: m5.2xlarge (8 vCPU, 32GB) = $0.384/시간
- CPU 사용률: 20%, Memory 사용률: 80%
- After: r5.xlarge (4 vCPU, 32GB) = $0.252/시간
- 절감: 34% ($95/월)
```

### 12.2 Auto Scaling

#### Target Tracking 정책

```yaml
# CloudWatch 기반 Auto Scaling
ScalingPolicy:
  PolicyType: TargetTrackingScaling
  TargetTrackingConfiguration:
    PredefinedMetricType: ASGAverageCPUUtilization
    TargetValue: 70.0  # CPU 70% 유지

  # 스케일 아웃: CPU > 70%
  ScaleOutCooldown: 60   # 60초 대기 후 추가 확장

  # 스케일 인: CPU < 70%
  ScaleInCooldown: 300   # 5분 대기 후 축소 (신중하게)

인스턴스 수 변화:
- 평상시 (CPU 30%): 2대
- 트래픽 증가 (CPU 80%): 4대로 자동 증가
- 트래픽 감소 (CPU 40%): 2대로 자동 축소

비용 절감:
- 고정 10대: $3,840/월
- Auto Scaling (평균 4대): $1,536/월 (60% 절감)
```

### 12.3 예약 인스턴스 & Spot 인스턴스

```
1. On-Demand (기본):
   - 가격: $0.096/시간
   - 유연성: 언제든 시작/종료
   - 사용처: 개발/테스트

2. Reserved Instance (1년 약정):
   - 가격: $0.062/시간 (35% 할인)
   - 3년 약정: $0.045/시간 (53% 할인)
   - 사용처: 프로덕션 베이스 용량

3. Spot Instance (경매):
   - 가격: $0.029/시간 (70% 할인)
   - 단점: AWS가 회수 가능 (2분 전 통지)
   - 사용처: 배치 작업, 스케일 아웃 보조

최적 조합:
- 베이스 용량 (항상 필요): Reserved 2대
- 변동 용량 (피크 시): Spot 0~5대 (Auto Scaling)
- 안전 버퍼: On-Demand 1대

월 비용 (평균 4대 운영):
- 전부 On-Demand: $277
- 혼합 (Reserved 2 + Spot 2): $134 (52% 절감)
```

---

## 13. CI/CD 최적화

### 13.1 빌드 속도 개선

#### Docker 레이어 캐싱

```dockerfile
# 나쁜 예: 코드 변경마다 전체 재빌드
FROM openjdk:11
WORKDIR /app
COPY . .                          # 전체 복사 (의존성 + 소스코드)
RUN ./gradlew build               # 매번 의존성 다운로드
CMD ["java", "-jar", "app.jar"]

# 빌드 시간: 5분 (매번)

# 좋은 예: 의존성 레이어 분리
FROM openjdk:11
WORKDIR /app

# 1. 의존성 파일만 먼저 복사
COPY build.gradle settings.gradle gradlew ./
COPY gradle ./gradle
RUN ./gradlew dependencies        # 의존성 다운로드 (캐시됨)

# 2. 소스코드 복사 및 빌드
COPY src ./src
RUN ./gradlew build

CMD ["java", "-jar", "build/libs/app.jar"]

# 빌드 시간:
# - 첫 빌드: 5분
# - 이후 빌드 (코드만 변경): 30초 (10배 빠름)
```

#### Gradle 빌드 캐시

```groovy
// build.gradle
buildCache {
    local {
        enabled = true
    }
    remote(HttpBuildCache) {
        url = 'https://build-cache.example.com'
        push = true  // CI에서 캐시 업로드
        credentials {
            username = System.getenv('CACHE_USERNAME')
            password = System.getenv('CACHE_PASSWORD')
        }
    }
}

성능 개선:
- 로컬 캐시: 30% 빌드 시간 단축
- 원격 캐시: 팀 전체 공유 → 50% 단축
```

### 13.2 병렬 빌드

#### GitHub Actions 병렬 Job

```yaml
# .github/workflows/ci.yml
name: CI

on: [push]

jobs:
  # Job 1: 테스트 (병렬)
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        test-suite: [unit, integration, e2e]
    steps:
      - uses: actions/checkout@v3
      - name: Run ${{ matrix.test-suite }} tests
        run: ./gradlew ${{ matrix.test-suite }}Test

  # Job 2: 린트 (병렬)
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run linter
        run: ./gradlew ktlintCheck

  # Job 3: 빌드 (테스트 + 린트 완료 후)
  build:
    needs: [test, lint]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build
        run: ./gradlew build

빌드 시간:
- 순차 실행: unit(5분) + integration(3분) + e2e(2분) + lint(1분) = 11분
- 병렬 실행: max(5분, 3분, 2분, 1분) = 5분 (2.2배 빠름)
```

### 13.3 배포 전략

#### Blue-Green 배포

```
현재 상태:
- Blue 환경: 프로덕션 (v1.0)
- Green 환경: 유휴

배포 프로세스:
1. Green 환경에 v1.1 배포
2. Green 환경 테스트 (헬스 체크, 스모크 테스트)
3. 로드 밸런서를 Blue → Green으로 전환
4. Blue 환경 대기 (롤백용)
5. 문제 없으면 Blue 환경 종료

장점:
- 무중단 배포
- 즉시 롤백 가능 (로드 밸런서만 전환)
- 프로덕션 환경 테스트

단점:
- 2배 리소스 필요 (비용)
```

#### Canary 배포

```yaml
# Istio VirtualService 예시
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: app-service
spec:
  hosts:
  - app.example.com
  http:
  - match:
    - headers:
        user-agent:
          regex: ".*iPhone.*"  # 특정 사용자만
    route:
    - destination:
        host: app-v2
  - route:
    - destination:
        host: app-v1
      weight: 90  # 90% 트래픽
    - destination:
        host: app-v2
      weight: 10  # 10% 트래픽 (Canary)

단계별 배포:
1. v2에 10% 트래픽
2. 모니터링 (에러율, 응답 시간)
3. 이상 없으면 → 50%
4. 최종 → 100%

장점:
- 점진적 배포 (위험 최소화)
- 실제 사용자 피드백
```

---

## 14. 모니터링 및 APM

### 14.1 메트릭 수집

#### Prometheus + Grafana

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'spring-boot'
    metrics_path: '/actuator/prometheus'
    static_configs:
      - targets: ['localhost:8080']

수집 메트릭:
- JVM: heap, thread, GC
- HTTP: 요청 수, 응답 시간, 에러율
- DB: 커넥션 풀 사용률, 쿼리 시간
- Custom: 비즈니스 메트릭 (주문 수, 결제 금액)
```

```java
// Spring Boot Actuator + Micrometer
@RestController
public class OrderController {

    private final MeterRegistry meterRegistry;

    @PostMapping("/orders")
    public Order createOrder(@RequestBody OrderRequest request) {
        Timer.Sample sample = Timer.start(meterRegistry);

        try {
            Order order = orderService.create(request);

            // 비즈니스 메트릭 기록
            meterRegistry.counter("orders.created",
                "status", order.getStatus()).increment();

            return order;
        } finally {
            sample.stop(Timer.builder("orders.create.time")
                .register(meterRegistry));
        }
    }
}
```

#### Grafana 대시보드

```
패널 1: 요청 처리량
Query: rate(http_requests_total[5m])
시각화: 시계열 그래프

패널 2: 응답 시간 (P50, P95, P99)
Query:
  - histogram_quantile(0.5, http_request_duration_seconds_bucket)
  - histogram_quantile(0.95, http_request_duration_seconds_bucket)
  - histogram_quantile(0.99, http_request_duration_seconds_bucket)

패널 3: 에러율
Query: rate(http_requests_total{status="500"}[5m]) / rate(http_requests_total[5m])

알림 설정:
- 에러율 > 1%: Slack 알림
- P95 응답 시간 > 1초: PagerDuty 호출
```

### 14.2 분산 추적 (Distributed Tracing)

#### Zipkin / Jaeger

```yaml
# application.yml
spring:
  sleuth:
    sampler:
      probability: 1.0  # 100% 트레이싱 (프로덕션은 0.1 = 10%)
  zipkin:
    base-url: http://zipkin:9411
```

```java
@RestController
public class OrderController {

    @Autowired
    private RestTemplate restTemplate;  // Sleuth가 자동으로 계측

    @GetMapping("/orders/{id}")
    public Order getOrder(@PathVariable Long id) {
        // 1. 주문 조회 (Span 1)
        Order order = orderRepository.findById(id).orElseThrow();

        // 2. 사용자 정보 조회 (Span 2)
        User user = restTemplate.getForObject(
            "http://user-service/users/" + order.getUserId(), User.class);

        // 3. 결제 정보 조회 (Span 3)
        Payment payment = restTemplate.getForObject(
            "http://payment-service/payments/" + order.getPaymentId(), Payment.class);

        order.setUser(user);
        order.setPayment(payment);
        return order;
    }
}

Zipkin에서 보이는 Trace:
┌────────────────────────────────────┐
│ GET /orders/123           [500ms]  │
├────────────────────────────────────┤
│  ├─ OrderRepository      [50ms]    │
│  ├─ GET /users/456       [200ms]   │  ← 병목 발견!
│  └─ GET /payments/789    [150ms]   │
└────────────────────────────────────┘

분석:
- 전체 응답 시간: 500ms
- User Service가 40% 차지 → 최적화 대상
```

### 14.3 로그 집계

#### ELK Stack (Elasticsearch + Logstash + Kibana)

```yaml
# logstash.conf
input {
  file {
    path => "/var/log/app/*.log"
    start_position => "beginning"
  }
}

filter {
  grok {
    match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}" }
  }
  date {
    match => [ "timestamp", "ISO8601" ]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "app-logs-%{+YYYY.MM.dd}"
  }
}
```

#### 로그 검색 예시 (Kibana)

```
쿼리 1: 5xx 에러 검색
level: ERROR AND status: 5*

쿼리 2: 느린 쿼리 검색
message: "SlowQuery" AND duration: > 1000

쿼리 3: 특정 사용자 액션 추적
user_id: 12345 AND timestamp: [2025-11-13T00:00:00 TO 2025-11-13T23:59:59]
```

### 14.4 실시간 알림

#### AlertManager (Prometheus)

```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'slack'

receivers:
- name: 'slack'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/XXX'
    channel: '#alerts'
    title: '{{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

# prometheus-alerts.yml
groups:
- name: performance
  rules:
  - alert: HighResponseTime
    expr: http_request_duration_seconds{quantile="0.99"} > 1
    for: 5m
    annotations:
      summary: "P99 응답 시간 > 1초"
      description: "{{ $labels.instance }}에서 5분간 P99 > 1초"

  - alert: HighErrorRate
    expr: rate(http_requests_total{status="500"}[5m]) / rate(http_requests_total[5m]) > 0.01
    for: 2m
    annotations:
      summary: "에러율 > 1%"
      description: "2분간 에러율 {{ $value }}%"
```

---

## 요약: 튜닝 우선순위

### 1단계: 빠른 효과 (Low Hanging Fruit)

```
✓ DB 인덱스 추가 (슬로우 쿼리 분석)
✓ N+1 쿼리 제거 (Fetch Join)
✓ 캐시 도입 (Redis, CDN)
✓ 로그 레벨 조정 (DEBUG → INFO)
✓ 파일 디스크립터 제한 증가

예상 효과: 30~50% 성능 개선
투자 시간: 1~2일
```

### 2단계: 아키텍처 개선

```
✓ 동기 → 비동기 전환 (이벤트 드리븐)
✓ 로드 밸런싱 도입
✓ Stateless 전환 (세션 → JWT)
✓ 커넥션 풀 튜닝
✓ 트랜잭션 범위 최소화

예상 효과: 50~200% 성능 개선
투자 시간: 1~2주
```

### 3단계: 인프라 최적화

```
✓ OS 레벨 TCP 튜닝
✓ JVM GC 튜닝
✓ Auto Scaling 설정
✓ CDN 도입
✓ HTTP/2 적용

예상 효과: 20~40% 성능 개선
투자 시간: 1주
```

### 4단계: 고급 최적화

```
✓ 분산 시스템 (MSA)
✓ CQRS + Event Sourcing
✓ Read Replica 분리
✓ Sharding
✓ 멀티 리전 배포

예상 효과: 10배 이상 확장성
투자 시간: 수개월
```

---

## 체크리스트 요약

### WAS 튜닝
- [ ] Tomcat 스레드 풀 최적화 (maxThreads, minSpareThreads)
- [ ] TCP 백로그 큐 증가 (somaxconn, tcp_max_syn_backlog)
- [ ] 파일 디스크립터 제한 증가 (ulimit -n)

### DB/쿼리 튜닝
- [ ] 슬로우 쿼리 로그 분석
- [ ] N+1 쿼리 제거 (Fetch Join, @EntityGraph)
- [ ] 인덱스 최적화 (카디널리티, 복합 인덱스 순서)
- [ ] 커넥션 풀 사이즈 조정
- [ ] 쿼리 결과 캐싱 (Redis)

### 비동기 처리
- [ ] @Async 또는 CompletableFuture 활용
- [ ] 메시지 큐 도입 (Kafka, RabbitMQ)
- [ ] 이벤트 드리븐 아키텍처 전환

### 로드 밸런싱
- [ ] L4/L7 로드 밸런서 선택
- [ ] 헬스 체크 설정
- [ ] 알고리즘 선택 (Round Robin, Least Conn, IP Hash)

### 캐시 최적화
- [ ] 계층별 캐시 전략 (브라우저, CDN, Redis, DB)
- [ ] Cache-Aside vs Write-Through 선택
- [ ] 캐시 Stampede 방지 (분산 락)
- [ ] TTL 설정

### 네트워크 튜닝
- [ ] TCP 윈도우 크기 조정
- [ ] TCP 혼잡 제어 (BBR)
- [ ] HTTP/2 활성화
- [ ] Gzip/Brotli 압축

### JVM 튜닝
- [ ] 힙 메모리 크기 설정 (-Xms, -Xmx)
- [ ] GC 알고리즘 선택 (G1GC, ZGC)
- [ ] GC 로그 분석
- [ ] Metaspace 크기 조정

### 애플리케이션
- [ ] Connection Leak 방지 (try-with-resources)
- [ ] 트랜잭션 범위 최소화
- [ ] 세션 → JWT 전환
- [ ] 비동기 로깅

### 파일시스템/디스크
- [ ] noatime 옵션 설정
- [ ] tmpfs 활용
- [ ] 스트리밍 처리 (대용량 파일)

### 오브젝트 스토리지
- [ ] S3 멀티파트 업로드
- [ ] Transfer Acceleration
- [ ] CloudFront + S3 조합

### 프론트엔드
- [ ] Code Splitting
- [ ] Tree Shaking
- [ ] 이미지 최적화 (WebP, Lazy Loading)
- [ ] React.memo / useMemo

### 클라우드
- [ ] 인스턴스 타입 최적화
- [ ] Auto Scaling 설정
- [ ] Reserved/Spot 인스턴스 활용

### CI/CD
- [ ] Docker 레이어 캐싱
- [ ] 빌드 캐시 (Gradle, Maven)
- [ ] 병렬 빌드
- [ ] Blue-Green / Canary 배포

### 모니터링
- [ ] Prometheus + Grafana
- [ ] 분산 추적 (Zipkin, Jaeger)
- [ ] 로그 집계 (ELK)
- [ ] 실시간 알림 (AlertManager)

---

**이 가이드는 실무에서 즉시 적용 가능한 성능 튜닝 체크리스트입니다.**
**각 섹션의 Before/After 비교를 참고하여 우선순위를 정하고 단계별로 적용하세요.**
