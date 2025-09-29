
# 10-Day Coding Interview Prep – Concepts, Problems & Python Solutions

> Language: **Python 3**  
> Focus: clean patterns + optimal solutions with brief proofs/intuition.  
> Format per day: **Concepts → Problems → Solutions (with explanations).**  
> Where helpful, LeetCode links are provided for practice.

---

## Day 1 – Big-O, Arrays & Strings Foundations
### Concepts
- Big-O refresher (time/space), common growth rates.
- Hashing for membership/counting (`dict`, `Counter`).
- String manipulation & two pointers.
- Test strategy: small cases, edge cases (empty, single, duplicates, negatives).

### Problems
1) **Two Sum (unsorted, O(n))** – Hash map of value→index.  
2) **Valid Anagram** – Frequency counting.  
3) **Longest Palindromic Substring** – Expand around center (O(n²), constant space).  

### Solutions
#### 1) Two Sum
```python
from typing import List, Tuple, Optional

def two_sum(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return seen[target - x], i
        seen[x] = i
    return None

# Explanation: For each x, we check if (target-x) was seen; if so, found pair. O(n) time, O(n) space.
```

#### 2) Valid Anagram
```python
from collections import Counter

def is_anagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)

# Explanation: Count characters and compare maps. Alternatively manual 26-array for lowercase only.
```

#### 3) Longest Palindromic Substring (Expand Around Center)
```python
def longest_palindrome(s: str) -> str:
    if not s: return ""
    res_l = res_r = 0

    def expand(l: int, r: int):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return l+1, r-1  # inclusive bounds

    for i in range(len(s)):
        l1, r1 = expand(i, i)       # odd length
        l2, r2 = expand(i, i+1)     # even length
        if r1 - l1 > res_r - res_l: res_l, res_r = l1, r1
        if r2 - l2 > res_r - res_l: res_l, res_r = l2, r2
    return s[res_l:res_r+1]

# Explanation: Each index is a center (odd/even). Expanding takes O(n) per center worst-case → O(n²) total.
```

---

## Day 2 – Sliding Window (Strings)
### Concepts
- Sliding window grow/shrink with two pointers.
- Window validity tracked by counts/sets.
- Typical questions: unique chars, covering all chars, permutations.

### Problems
1) **Longest Substring Without Repeating Characters** – set-based window.  
2) **Minimum Window Substring** – window covers target multiset.  
3) **Permutation in String** – fixed window + counts comparison.

### Solutions
#### 1) Longest Substring Without Repeating
```python
def length_of_longest_substring(s: str) -> int:
    seen = set()
    l = ans = 0
    for r, ch in enumerate(s):
        while ch in seen:
            seen.remove(s[l]); l += 1
        seen.add(ch)
        ans = max(ans, r - l + 1)
    return ans

# Explanation: Maintain a window with unique chars; shrink until no duplicate.
```

#### 2) Minimum Window Substring
```python
from collections import Counter

def min_window(s: str, t: str) -> str:
    if len(t) > len(s): return ""
    need = Counter(t)
    need_total = len(t)
    l = 0
    best = (float("inf"), 0, 0)

    for r, ch in enumerate(s):
        if need[ch] > 0:
            need_total -= 1
        need[ch] -= 1

        while need_total == 0:
            if r - l + 1 < best[0]:
                best = (r - l + 1, l, r + 1)
            left_ch = s[l]
            need[left_ch] += 1
            if need[left_ch] > 0:
                need_total += 1
            l += 1

    return "" if best[0] == float("inf") else s[best[1]:best[2]]

# Explanation: Expand until all t covered, then shrink to minimal; track best window.
```

#### 3) Permutation in String
```python
from collections import Counter

def check_inclusion(p: str, s: str) -> bool:
    if len(p) > len(s): return False
    need = Counter(p)
    window = Counter(s[:len(p)])
    if window == need: return True
    for i in range(len(p), len(s)):
        window[s[i]] += 1
        left = s[i - len(p)]
        window[left] -= 1
        if window[left] == 0:
            del window[left]
        if window == need:
            return True
    return False

# Explanation: Maintain counts for a fixed-size window of |p| and compare.
```

---

## Day 3 – Arrays & Prefix Sums
### Concepts
- Prefix sum (1D), HashMap of prefix frequencies.
- Convert problems with transformations (e.g., 0→-1).

### Problems
1) **Subarray Sum Equals K** – count subarrays with sum k.  
2) **Contiguous Array** – longest subarray with equal 0/1.  
3) **Product of Array Except Self** – prefix & suffix products.  
4) **Range Sum Query (Immutable)** – prefix array.

### Solutions
```python
def subarray_sum(nums, k):
    freq = {0: 1}
    s = ans = 0
    for x in nums:
        s += x
        ans += freq.get(s - k, 0)
        freq[s] = freq.get(s, 0) + 1
    return ans
# Explanation: For each prefix s, #subarrays ending here with sum k is count of (s-k) seen.

def find_max_length(nums):
    first = {0: -1}
    s = ans = 0
    for i, x in enumerate(nums):
        s += 1 if x == 1 else -1
        if s in first:
            ans = max(ans, i - first[s])
        else:
            first[s] = i
    return ans
# Explanation: Equal 0/1 ↔ prefix sums equal; store first index per sum.

def product_except_self(nums):
    n = len(nums)
    res = [1]*n
    pref = 1
    for i in range(n):
        res[i] = pref
        pref *= nums[i]
    suf = 1
    for i in range(n-1, -1, -1):
        res[i] *= suf
        suf *= nums[i]
    return res
# Explanation: res[i] = product(left of i) * product(right of i).

class NumArray:
    def __init__(self, nums):
        self.pref = [0]
        for x in nums:
            self.pref.append(self.pref[-1] + x)
    def sumRange(self, l, r):
        return self.pref[r+1] - self.pref[l]
# Explanation: Prefix makes range queries O(1).
```

---

## Day 4 – Two Pointers & Binary Search (Arrays)
### Concepts
- Two pointers on sorted arrays, partitioning.
- Binary search variants, rotated arrays.

### Problems & Solutions
```python
# Two Sum II (sorted)
def two_sum_sorted(nums, target):
    l, r = 0, len(nums)-1
    while l < r:
        s = nums[l] + nums[r]
        if s == target: return [l, r]
        if s < target: l += 1
        else: r -= 1
    return []

# 3Sum (sorted + two pointers)
def three_sum(nums):
    nums.sort()
    n, res = len(nums), []
    for i in range(n-2):
        if i and nums[i] == nums[i-1]: continue
        l, r = i+1, n-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                res.append([nums[i], nums[l], nums[r]])
                l += 1; r -= 1
                while l < r and nums[l] == nums[l-1]: l += 1
                while l < r and nums[r] == nums[r+1]: r -= 1
            elif s < 0: l += 1
            else: r -= 1
    return res

# Container With Most Water
def max_area(h):
    l, r = 0, len(h)-1
    ans = 0
    while l < r:
        ans = max(ans, (r-l)*min(h[l], h[r]))
        if h[l] < h[r]: l += 1
        else: r -= 1
    return ans

# Search in Rotated Sorted Array
def search_rotated(nums, target):
    l, r = 0, len(nums)-1
    while l <= r:
        m = (l+r)//2
        if nums[m] == target: return m
        if nums[l] <= nums[m]:
            if nums[l] <= target < nums[m]: r = m-1
            else: l = m+1
        else:
            if nums[m] < target <= nums[r]: l = m+1
            else: r = m-1
    return -1
```

---

## Day 5 – Linked Lists
### Concepts
- Pointers manipulation, dummy nodes.
- Fast/slow pointers (cycle detection), in-place reversal.

### Problems & Solutions
```python
class ListNode:
    def __init__(self, val=0, next=None): self.val = val; self.next = next

# Reverse Linked List
def reverse_list(head):
    prev, cur = None, head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev

# Merge Two Sorted Lists
def merge_two_lists(l1, l2):
    dummy = tail = ListNode(0)
    while l1 and l2:
        if l1.val < l2.val:
            tail.next, l1 = l1, l1.next
        else:
            tail.next, l2 = l2, l2.next
        tail = tail.next
    tail.next = l1 or l2
    return dummy.next

# Detect Cycle (Floyd)
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast: return True
    return False

# Reorder List (L0→Ln→L1→Ln-1...)
def reorder_list(head):
    if not head or not head.next: return
    # find mid
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next; fast = fast.next.next
    # reverse second half
    prev, cur = None, slow.next
    slow.next = None
    while cur:
        nxt = cur.next; cur.next = prev; prev = cur; cur = nxt
    # merge
    p1, p2 = head, prev
    while p2:
        t1, t2 = p1.next, p2.next
        p1.next = p2; p2.next = t1
        p1, p2 = t1, t2
```

---

## Day 6 – Stacks, Queues & Monotonic Structures
### Concepts
- Stack for validation, monotonic stack for next greater/prev smaller.
- Deque for sliding window maximum.

### Problems & Solutions
```python
# Valid Parentheses
def is_valid(s: str) -> bool:
    stack, pairs = [], {')':'(', ']':'[', '}':'{'}
    for ch in s:
        if ch in pairs.values(): stack.append(ch)
        else:
            if not stack or stack[-1] != pairs.get(ch, None): return False
            stack.pop()
    return not stack

# Min Stack
class MinStack:
    def __init__(self): self.st = []; self.min_st = []
    def push(self, x):
        self.st.append(x)
        self.min_st.append(x if not self.min_st else min(x, self.min_st[-1]))
    def pop(self): self.st.pop(); self.min_st.pop()
    def top(self): return self.st[-1]
    def getMin(self): return self.min_st[-1]

# Daily Temperatures (Monotonic decreasing stack)
def daily_temperatures(T):
    res = [0]*len(T)
    st = []  # indices stack
    for i, t in enumerate(T):
        while st and T[st[-1]] < t:
            j = st.pop()
            res[j] = i - j
        st.append(i)
    return res

# Sliding Window Maximum (Deque)
from collections import deque
def max_sliding_window(nums, k):
    dq = deque()
    res = []
    for i, x in enumerate(nums):
        while dq and dq[0] <= i-k: dq.popleft()
        while dq and nums[dq[-1]] <= x: dq.pop()
        dq.append(i)
        if i >= k-1: res.append(nums[dq[0]])
    return res
```

---

## Day 7 – Binary Trees (DFS/BFS)
### Concepts
- Recursion patterns, traversal orders.
- BFS for level-order, DFS for path properties.

### Problems & Solutions
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

# Inorder Traversal (iterative)
def inorder_traversal(root):
    res, st = [], []
    cur = root
    while cur or st:
        while cur:
            st.append(cur); cur = cur.left
        cur = st.pop()
        res.append(cur.val)
        cur = cur.right
    return res

# Level Order Traversal (BFS)
from collections import deque
def level_order(root):
    if not root: return []
    res, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level)
    return res

# Diameter of Binary Tree
def diameter_of_binary_tree(root):
    ans = 0
    def depth(node):
        nonlocal ans
        if not node: return 0
        l = depth(node.left); r = depth(node.right)
        ans = max(ans, l + r)
        return 1 + max(l, r)
    depth(root)
    return ans

# Lowest Common Ancestor (BST or BT—here BT)
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q: return root
    L = lowest_common_ancestor(root.left, p, q)
    R = lowest_common_ancestor(root.right, p, q)
    if L and R: return root
    return L or R
```

---

## Day 8 – Graphs (BFS/DFS, Toposort, Union-Find)
### Concepts
- Graph traversal, visited sets.
- Topological sort for DAG prerequisites.
- Union-Find for connectivity/cycles.

### Problems & Solutions
```python
# Clone Graph (DFS)
def cloneGraph(node):
    if not node: return None
    mp = {}
    def dfs(n):
        if n in mp: return mp[n]
        copy = type(n)(n.val, [])
        mp[n] = copy
        for nei in n.neighbors:
            copy.neighbors.append(dfs(nei))
        return copy
    return dfs(node)

# Number of Islands (DFS/BFS on grid)
def num_islands(grid):
    if not grid: return 0
    m, n = len(grid), len(grid[0])
    def dfs(i, j):
        if i<0 or j<0 or i>=m or j>=n or grid[i][j] != '1': return
        grid[i][j] = '#'
        dfs(i+1,j); dfs(i-1,j); dfs(i,j+1); dfs(i,j-1)
    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1; dfs(i, j)
    return count

# Course Schedule (Topo Sort - Kahn)
from collections import deque, defaultdict
def can_finish(numCourses, prerequisites):
    indeg = [0]*numCourses
    adj = defaultdict(list)
    for a,b in prerequisites:
        adj[b].append(a); indeg[a]+=1
    q = deque([i for i in range(numCourses) if indeg[i]==0])
    seen = 0
    while q:
        u = q.popleft(); seen += 1
        for v in adj[u]:
            indeg[v]-=1
            if indeg[v]==0: q.append(v)
    return seen == numCourses

# Union-Find (Disjoint Set)
class DSU:
    def __init__(self, n):
        self.p = list(range(n)); self.r = [0]*n
    def find(self, x):
        if self.p[x]!=x: self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa==pb: return False
        if self.r[pa]<self.r[pb]: pa, pb = pb, pa
        self.p[pb]=pa
        if self.r[pa]==self.r[pb]: self.r[pa]+=1
        return True

def find_redundant_connection(edges):
    n = len(edges)
    dsu = DSU(n+1)
    for u, v in edges:
        if not dsu.union(u, v):
            return [u, v]
    return []
```

---

## Day 9 – Dynamic Programming (1D/2D)
### Concepts
- Bottom-up tabulation; state definition & transitions.
- Common patterns: Knapsack/Subset, Edit distance, LIS.

### Problems & Solutions
```python
# House Robber (1D DP)
def rob(nums):
    prev2 = prev1 = 0
    for x in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + x)
    return prev1

# Coin Change (Min coins for amount)
def coin_change(coins, amount):
    INF = amount+1
    dp = [INF]*(amount+1); dp[0]=0
    for c in coins:
        for a in range(c, amount+1):
            dp[a] = min(dp[a], dp[a-c]+1)
    return dp[amount] if dp[amount] != INF else -1

# Longest Increasing Subsequence (O(n log n))
import bisect
def length_of_lis(nums):
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails): tails.append(x)
        else: tails[i] = x
    return len(tails)

# Edit Distance (Levenshtein)
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```

---

## Day 10 – Backtracking & Search
### Concepts
- Build partial solutions; choose/explore/unchoose.
- Pruning by constraints.

### Problems & Solutions
```python
# Subsets
def subsets(nums):
    res, path = [], []
    def dfs(i):
        if i == len(nums):
            res.append(path[:]); return
        dfs(i+1)
        path.append(nums[i]); dfs(i+1); path.pop()
    dfs(0); return res

# Permutations
def permute(nums):
    res = []
    def backtrack(i):
        if i == len(nums):
            res.append(nums[:]); return
        for j in range(i, len(nums)):
            nums[i], nums[j] = nums[j], nums[i]
            backtrack(i+1)
            nums[i], nums[j] = nums[j], nums[i]
    backtrack(0); return res

# Combination Sum
def combination_sum(candidates, target):
    res, path = [], []
    candidates.sort()
    def dfs(i, t):
        if t == 0: res.append(path[:]); return
        if t < 0 or i == len(candidates): return
        # choose i
        path.append(candidates[i])
        dfs(i, t - candidates[i])
        path.pop()
        # skip i
        dfs(i+1, t)
    dfs(0, target); return res

# N-Queens (layouts)
def solve_n_queens(n):
    res = []
    cols, d1, d2 = set(), set(), set()
    board = [["."]*n for _ in range(n)]
    def dfs(r):
        if r == n:
            res.append(["".join(row) for row in board]); return
        for c in range(n):
            if c in cols or (r-c) in d1 or (r+c) in d2: continue
            cols.add(c); d1.add(r-c); d2.add(r+c)
            board[r][c] = "Q"
            dfs(r+1)
            board[r][c] = "."
            cols.remove(c); d1.remove(r-c); d2.remove(r+c)
    dfs(0); return res

# Word Search
def exist(board, word):
    m, n = len(board), len(board[0])
    used = [[False]*n for _ in range(m)]
    def dfs(i, j, k):
        if k == len(word): return True
        if i<0 or j<0 or i>=m or j>=n or used[i][j] or board[i][j] != word[k]:
            return False
        used[i][j] = True
        ok = dfs(i+1,j,k+1) or dfs(i-1,j,k+1) or dfs(i,j+1,k+1) or dfs(i,j-1,k+1)
        used[i][j] = False
        return ok
    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0): return True
    return False
```

---

## Appendix – Quick Patterns & Tips
- **Sliding window:** Expand right, shrink left while maintaining invariants.
- **Prefix sum:** Convert range/sum counting to O(n) with hash maps.
- **Monotonic stack/deque:** Next greater elements, window maxima in O(n).
- **Fast/slow pointers:** Cycle detection, mid finding.
- **Binary search on answer:** When function is monotonic (e.g., capacity, time).
- **DP mantra:** Define state, transition, base cases; check overlapping subproblems.
- **Backtracking:** Choose → explore → un-choose; prune aggressively.

If you reached here just do remember that I am proud of you, always was proud of you
You can rock it ji,
