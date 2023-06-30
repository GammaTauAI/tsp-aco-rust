use std::{
    collections::{HashMap, HashSet, VecDeque},
    ops::Range,
};

use clap::Parser;
use lazy_static::lazy_static;
use petgraph::{graph::NodeIndex, prelude::UnGraph, Graph};
use piston_window::{PistonWindow, WindowSettings};
use plotters::prelude::*;
use plotters_piston::{draw_piston_window, PistonBackend};
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng};

// prints only if #[cfg(debug_assertions)] is set
macro_rules! debug {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            println!($($arg)*);
        }
    };
}

#[derive(Parser)]
struct Args {
    #[clap(short, long, default_value = "60")]
    fps: u32,
    #[clap(short, long, default_value = "50")]
    epochs: usize,
    #[clap(short, long, default_value = "20")]
    points: usize,
    #[clap(short, long, default_value = "3")]
    ants: usize,
    #[clap(long, default_value = "1.0")]
    alpha: f64,
    #[clap(long, default_value = "1.0")]
    beta: f64,
    #[clap(long, default_value = "0.5")]
    eva: f64,
    #[clap(long, default_value = "1")]
    q: f64,
    #[clap(short, long, default_value_t = false)]
    instant_epoch: bool,
    #[clap(long, default_value = "shortest")]
    retrieval: String,
}

/// i know, i know, this is a terrible way to do this
fn sample_range_with_resampling(points: &VecDeque<(i32, i32)>, range: Range<i32>) -> (i32, i32) {
    let mut rng = rand::thread_rng();
    'outer: loop {
        let random_x = rng.gen_range(range.clone());
        let random_y = rng.gen_range(range.clone());
        for (x, y) in points.iter() {
            if (x - random_x).abs() < 40 && (y - random_y).abs() < 40 {
                continue 'outer;
            }
        }
        return (random_x, random_y);
    }
}

#[derive(Debug, Copy, Clone)]
struct NodeData {
    x: i32,
    y: i32,
}
#[derive(Debug, Copy, Clone, Default)]
struct EdgeData {
    dist: f64,
    pheromone: f64,
}

fn is_fully_connected(graph: &UnGraph<NodeData, EdgeData>) -> bool {
    let nodes = graph.node_indices();
    for node in nodes.clone() {
        let count = graph.neighbors(node).count();
        if count != nodes.clone().count() - 1 {
            return false;
        }
    }
    true
}

fn initialize_graph(points: &VecDeque<(i32, i32)>) -> UnGraph<NodeData, EdgeData> {
    let mut graph = Graph::new_undirected();
    let mut node_map = std::collections::HashMap::new();
    for (x, y) in points.iter() {
        let node = graph.add_node(NodeData { x: *x, y: *y });
        node_map.insert((x, y), node);
    }

    for (i, (x1, y1)) in points.iter().enumerate() {
        for (x2, y2) in points.iter().skip(i + 1) {
            if (x1, y1) != (x2, y2) {
                let dist = (((x1 - x2).pow(2) + (y1 - y2).pow(2)) as f64).sqrt();
                graph.add_edge(
                    *node_map.get(&(x1, y1)).unwrap(),
                    *node_map.get(&(x2, y2)).unwrap(),
                    EdgeData {
                        dist,
                        ..Default::default()
                    },
                );
            }
        }
    }
    assert!(is_fully_connected(&graph)); // making sure that the graph is fully connected
    graph
}

lazy_static! {
    static ref ARGS: Args = Args::parse();
}

const WINDOW_TITLE: &str = "Traveling Salesman Problem";

#[derive(Debug, Clone)]
struct Ant {
    path: Vec<NodeIndex>,
    path_length: f64,
    visited: HashSet<NodeIndex>,
    iter: usize,
}

impl Ant {
    fn new(start_node: NodeIndex) -> Self {
        Self {
            path: vec![start_node],
            path_length: 0.0,
            visited: HashSet::from([start_node]),
            iter: 0,
        }
    }

    fn advance(&mut self, params: &ACOParams, graph: &UnGraph<NodeData, EdgeData>) -> NodeIndex {
        debug!("path: {:?}", self.path.len());
        // if last node, return to start
        if self.path.len() == graph.node_count() {
            self.path.push(self.path[0]);
            return self.path[0];
        }
        let current_node = self.path.last().unwrap();
        let mut rng = rand::thread_rng();
        let mut probs = Vec::new();
        let mut weis = HashMap::new();
        let mut wei_sum = 0.0;
        let neighbors = graph
            .neighbors(*current_node)
            .filter(|n| !self.visited.contains(n))
            .collect::<Vec<_>>();

        // TODO: can merge these two loops

        for neighbor in &neighbors {
            let edge = graph.find_edge(*current_node, *neighbor).unwrap();
            let edge_data = graph.edge_weight(edge).unwrap();
            // hack to avoid div by zero.
            let pheromone = if edge_data.pheromone == 0.0 {
                0.0001
            } else {
                edge_data.pheromone
            };
            let wei = pheromone.powf(params.alpha) * (1.0 / edge_data.dist).powf(params.beta);
            debug!("wei: {}", wei);
            weis.insert(neighbor, wei);
            wei_sum += wei;
        }

        for neighbor in &neighbors {
            let wei = weis.get(neighbor).unwrap();
            let prob = wei / wei_sum;
            probs.push(prob);
        }

        // pick a random neighbor based on the probabilities
        debug!("probs: {:?}", probs);
        let dist = WeightedIndex::new(&probs).unwrap();
        let pick = dist.sample(&mut rng);
        let next_node = neighbors[pick];
        self.path_length += graph
            .edge_weight(graph.find_edge(*current_node, next_node).unwrap())
            .unwrap()
            .dist;
        debug!("pick {}: {}", probs[pick], pick);
        self.path.push(next_node);
        self.visited.insert(next_node);
        next_node
    }

    fn deposit_pheromones(&self, graph: &mut UnGraph<NodeData, EdgeData>) {
        for (i, node) in self.path.iter().enumerate() {
            if i == self.path.len() - 1 {
                break;
            }
            let next_node = self.path[i + 1];
            let edge = graph.find_edge(*node, next_node).unwrap();
            let edge_data = graph.edge_weight_mut(edge).unwrap();
            edge_data.pheromone += ARGS.q / self.path_length;
        }
    }

    fn next_iter(&mut self) {
        let start = self.path[0];
        self.path = vec![start];
        self.path_length = 0.0;
        self.visited = HashSet::from([start]);
        self.iter += 1;
    }
}

#[derive(Debug, Clone)]
struct ACOParams {
    alpha: f64,
    beta: f64,
    eva: f64,
}

#[derive(Debug, Clone)]
struct ACOState {
    graph: UnGraph<NodeData, EdgeData>,
    params: ACOParams,
    ants: Vec<Ant>,
}

impl ACOState {
    fn advance_ants(&mut self) -> (Vec<(&NodeData, &NodeData)>, bool) {
        let start_node = self.ants[0].path[0];
        let mut advancements = Vec::new();
        let mut is_finished = false;
        for ant in &mut self.ants {
            let current_node = ant.path.last().unwrap().to_owned();
            let next_node = ant.advance(&self.params, &self.graph);
            advancements.push((
                self.graph.node_weight(current_node).unwrap(),
                self.graph.node_weight(next_node).unwrap(),
            ));
            if next_node == start_node {
                is_finished = true;
            }
        }
        (advancements, is_finished)
    }

    fn update_pheromones(&mut self) {
        for edge in self.graph.edge_indices() {
            let edge_data = self.graph.edge_weight_mut(edge).unwrap();
            edge_data.pheromone *= self.params.eva;
        }
        for ant in &mut self.ants {
            ant.deposit_pheromones(&mut self.graph);
            ant.next_iter();
        }
    }

    fn most_common_path(&self) -> (Vec<&NodeData>, f64) {
        let start = self.ants[0].path[0];
        let start_data = self.graph.node_weight(start).unwrap();
        let mut path = vec![start];
        let mut path_data = vec![start_data];
        let mut total_dist = 0.0;
        while path.len() < self.graph.node_count() {
            let mut node_occurences = HashMap::new();
            // TODO: there is an optimization that can be done here
            for ant in &self.ants {
                let next_node = ant.path[path.len()];
                let count = node_occurences.entry(next_node).or_insert(0);
                *count += 1;
            }
            let argmax = node_occurences
                .iter()
                .max_by(|(_, a), (_, b)| a.cmp(b))
                .unwrap()
                .0;
            let edge = self.graph.find_edge(start, *argmax).unwrap();
            let edge_data = self.graph.edge_weight(edge).unwrap();
            total_dist += edge_data.dist;
            path.push(*argmax);
            path_data.push(self.graph.node_weight(*argmax).unwrap());
        }
        path_data.push(start_data);
        (path_data, total_dist)
    }

    fn shortest_path(&self) -> (Vec<&NodeData>, f64) {
        let mut shortest_path = None;
        let mut shortest_dist = f64::INFINITY;
        for ant in &self.ants {
            let dist = ant.path_length;
            if dist < shortest_dist {
                shortest_dist = dist;
                shortest_path = Some(
                    ant.path
                        .iter()
                        .map(|n| self.graph.node_weight(*n).unwrap())
                        .collect::<Vec<_>>(),
                );
            }
        }
        (shortest_path.unwrap(), shortest_dist)
    }
}

#[derive(Debug)]
enum State {
    PointsGen,
    InitGraph,
    Advancing,
    PheroUpdate,
    NextIter,
    Done,
    Sleep,
}

fn main() {
    assert!(ARGS.ants > 0);
    assert!(ARGS.epochs > 0);
    assert!(ARGS.alpha > 0.0);
    assert!(ARGS.beta >= 1.0);

    let mut window: PistonWindow = WindowSettings::new(WINDOW_TITLE, [1100, 1100])
        .samples(4)
        .build()
        .unwrap();

    let ant_colors = (0..ARGS.ants).map(Palette99::pick).collect::<Vec<_>>();
    let mut points = VecDeque::new();
    let mut edge_lines = VecDeque::new();
    let mut dists_collect = Vec::new();

    let mut state = State::PointsGen;
    let mut aco = None;
    let mut epoch = 0usize;

    let mut f = |b: PistonBackend| {
        let root = b.into_drawing_area();
        root.fill(&WHITE)?;

        let mut add_line = |i: usize, from: &NodeData, to: &NodeData| {
            // offset by ant index to avoid overlapping lines
            let i_i32 = i as i32;
            let (x1, y1) = (from.x + i_i32, from.y + i_i32);
            let (x2, y2) = (to.x + i_i32, to.y + i_i32);
            edge_lines.push_back((&ant_colors[i], (x1, y1), (x2, y2)));
        };

        let mut cc = ChartBuilder::on(&root)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .caption(
                format!("{} Epoch: {}", WINDOW_TITLE, epoch),
                ("sans-serif", 40),
            )
            .build_cartesian_2d(-500..501, -500..501)?;

        cc.configure_mesh().draw()?;

        match &mut state {
            State::PointsGen => {
                let range = -500..501;
                let coord = sample_range_with_resampling(&points, range);
                points.push_back(coord);
                if points.len() == ARGS.points {
                    state = State::InitGraph;
                }
            }
            State::InitGraph => {
                let graph = initialize_graph(&points);
                let start_node = graph.node_indices().next().unwrap();
                aco = Some(ACOState {
                    graph,
                    params: ACOParams {
                        alpha: ARGS.alpha,
                        beta: ARGS.beta,
                        eva: ARGS.eva,
                    },
                    ants: vec![Ant::new(start_node); ARGS.ants],
                });

                state = State::Advancing;
            }
            State::Advancing => {
                let aco = aco.as_mut().unwrap();
                loop {
                    let (advancements, is_finished) = aco.advance_ants();
                    debug!("finished: {}", is_finished);
                    advancements
                        .into_iter()
                        .enumerate()
                        .for_each(|(i, (from, to))| {
                            add_line(i, from, to);
                        });
                    if is_finished {
                        if epoch != ARGS.epochs {
                            state = State::PheroUpdate;
                            epoch += 1;
                        } else {
                            state = State::Done;
                        }
                        break;
                    }
                    if !ARGS.instant_epoch {
                        break;
                    }
                }
            }
            State::PheroUpdate => {
                debug!("updating pheromones");
                let aco = aco.as_mut().unwrap();
                aco.update_pheromones();
                state = State::NextIter;
            }
            State::NextIter => {
                edge_lines.clear();
                state = State::Advancing;
            }
            State::Done => {
                let aco = aco.as_ref().unwrap();
                let (path, total_dist) = match ARGS.retrieval.as_str() {
                    "common" => aco.most_common_path(),
                    "shortest" => aco.shortest_path(),
                    _ => panic!("invalid retrieval method"),
                };
                dists_collect.push(total_dist);
                println!("total dist: {}", total_dist);
                println!("dists: {:?}", dists_collect);
                println!("path: {:?}", path);
                // https://www.colourlovers.com/color/E4FC5B/yellow_marker
                let yellow_marker = RGBAColor(228, 252, 91, 0.75).stroke_width(10);
                cc.draw_series(path.iter().take(path.len()).zip(path.iter().skip(1)).map(
                    |(from, to)| {
                        let xy1 = (from.x - 1, from.y - 1);
                        let xy2 = (to.x - 1, to.y - 1);
                        PathElement::new(vec![xy1, xy2], yellow_marker)
                    },
                ))
                .unwrap();
                state = State::Sleep;
            }
            State::Sleep => {
                // wait for input
                println!("Press enter to retry the same problem");
                std::io::stdin().read_line(&mut String::new()).unwrap();
                state = State::InitGraph;
                epoch = 0;
                edge_lines.clear();
            }
        }

        cc.draw_series(
            points
                .iter()
                .enumerate()
                .map(|(i, (x, y))| Circle::new((*x, *y), 10, if i == 0 { RED } else { BLUE })),
        )
        .unwrap();

        if ARGS.instant_epoch {
            return Ok(());
        }

        if !matches!(state, State::Sleep) {
            cc.draw_series(edge_lines.iter().map(|(color, (x1, y1), (x2, y2))| {
                PathElement::new(vec![(*x1, *y1), (*x2, *y2)], color.stroke_width(2))
            }))
            .unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(1000 / ARGS.fps as u64));
        Ok(())
    };
    while draw_piston_window(&mut window, &mut f).is_some() {}
}
