pub trait SegmentTreeSpec {
    type S: Clone;
    type F: Clone;
    
    /// Require for all a,b,c: op(a, op(b, c)) = op(op(a, b), c)
    fn op(a: &Self::S, b: &Self::S) -> Self::S;
    /// Require for all a: op(a, identity()) = op(identity(), a) = a
    fn identity() -> Self::S;
    /// For eager updates, compose() can be unimplemented!(). For lazy updates:
    /// Require for all f,g,a: apply(compose(f, g), a) = apply(f, apply(g, a))
    fn compose(f: &Self::F, g: &Self::F) -> Self::F;
    /// For eager updates, apply() can assume to act on a leaf. For lazy updates:
    /// Require for all f,a,b: apply(f, op(a, b)) = op(apply(f, a), apply(f, b))
    fn apply(f: &Self::F, a: &Self::S, size: i64) -> Self::S;
}

pub struct SegmentTree<T: SegmentTreeSpec> {
    size: usize,
    val: Vec<T::S>,
    app: Vec<Option<T::F>>,
}

impl<T: SegmentTreeSpec> SegmentTree<T> {
    pub fn new(val: Vec<T::S>) -> Self {
        let size = val.capacity();
        let mut tree_val = vec![T::identity(); size];
        tree_val.extend_from_slice(&val);

        for i in (1..size).rev() {
            tree_val[i] = T::op(&tree_val[2*i], &tree_val[2*i+1]);
        }

        let app = vec![None; size];

        Self{size, val:tree_val, app}
    }

    fn modify_node_lazily(&mut self, node_idx: usize, f: &T::F, segment_size: i64) {
        self.val[node_idx] = T::apply(f, &self.val[node_idx], segment_size);
        if node_idx < self.size {
            let h = match &self.app[node_idx] {
                Some(g) => T::compose(f, g),
                None => f.clone(),
            };
            self.app[node_idx] = Some(h);
        }
    }

    fn sift_up(&mut self, mut node_idx: usize, segment_size: i64) {
        while node_idx > 1 {
            node_idx = node_idx/2;
            match &self.app[node_idx] {
                Some(f) => self.val[node_idx] = T::apply(&f, &T::op(&self.val[2*node_idx], &self.val[2*node_idx + 1]), segment_size),
                None => self.val[node_idx] = T::op(&self.val[2*node_idx], &self.val[2*node_idx + 1]),
            }
        }
    }

    fn sift_down(&mut self, node_idx: usize) {
        let level = (node_idx as f32).log2().floor() as i32 + 1;

        for i in (1..level).rev() {
            let p = node_idx / ((i as f64).exp2() as usize);
            if let Some(ref f) = self.app[p].take() {
                let s = (i as f64).exp2() as i64; 
                self.modify_node_lazily(p * 2, f, s);
                self.modify_node_lazily(p * 2 + 1, f, s);
                self.app[p] = None;
            }
        }
    }

    pub fn update(&mut self, mut l: usize, mut r: usize, f: &T::F) {
        l += self.size;
        r += self.size;

        let l0 = l.clone();
        let r0 = r.clone();
        let mut s = 1;

        while l < r {
            if l % 2 == 1 {
                self.modify_node_lazily(l, f, s);
                l += 1;
            }
            if r % 2 == 1 {
                r-=1;
                self.modify_node_lazily(r, f, s);
            }

            s *= 2;
            l /= 2;
            r /= 2;
        }

        self.sift_up(l0, 1);
        self.sift_up(r0 - 1, 1);
    }
    
    pub fn query(&mut self, mut l: usize, mut r: usize) -> T::S {
        l += self.size;
        r += self.size;

        self.sift_down(l);
        self.sift_down(r - 1);

        let mut result = T::identity();
        while l < r {
            if l % 2 == 1 {
                result = T::op(&self.val[l], &result);
                l += 1;
            }
            if r % 2 == 1 {
                r -= 1;
                result = T::op(&self.val[r], &result);
            }

            l /= 2;
            r /= 2;
        }

        result
    }
}

mod tests {
    use crate::{SegmentTreeSpec, SegmentTree};

    pub enum AssignMax {}
    impl SegmentTreeSpec for AssignMax {
        type S = i64;
        type F = i64;
        fn op(&a: &Self::S, &b: &Self::S) -> Self::S {
            a.max(b)
        }
        fn identity() -> Self::S {
            i64::min_value()
        }
        fn compose(&f: &Self::F, g: &Self::F) -> Self::F {
            f + g
        }
        fn apply(&f: &Self::F, a: &Self::S, _: i64) -> Self::S {
            f + a
        }
    }

    #[test]
    fn test_eager() {
        let mut segment_tree:SegmentTree<AssignMax> = SegmentTree::new(vec![2,3,8,4,0,1,3,9]);
        segment_tree.update(4, 5, &12);
        assert_eq!(8, segment_tree.query(0, 4));
        assert_eq!(12, segment_tree.query(4, 8));
    }

    #[test]
    fn test_lazy() {
        let mut segment_tree:SegmentTree<AssignMax> = SegmentTree::new(vec![2,3,8,4,0,1,3,9]);
        segment_tree.update(2, 7, &12);
        assert_eq!(20, segment_tree.query(1, 3));
        assert_eq!(15, segment_tree.query(4, 7));
    }

    #[test]
    fn test_lazy_incomplete() {
        let mut segment_tree:SegmentTree<AssignMax> = SegmentTree::new(vec![2,3,8,4,0,1,3,9,10]);
        segment_tree.update(2, 7, &12);
        assert_eq!(20, segment_tree.query(1, 3));
        assert_eq!(15, segment_tree.query(4, 7));
        assert_eq!(10, segment_tree.query(7, 9));
    }

}