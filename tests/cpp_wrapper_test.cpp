#include <motionbricks/motionbricks.hpp>

#include <cassert>
#include <utility>

int main() {
    motionbricks::runtime_options options;
    assert(options.get() != nullptr);
    motionbricks::runtime_options moved(std::move(options));
    assert(options.get() == nullptr);
    assert(moved.get() != nullptr);
    return 0;
}
